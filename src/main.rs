use std::collections::HashMap;
use std::hash::Hash;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Context;
use candle_core::{DType, IndexOp, Tensor};
use embed_anything::embeddings::local::colsmol::ColSmolEmbedder;
use iced::Length::Fill;
use iced::keyboard::Key;
use iced::widget::combo_box;
use iced::widget::image::Handle;
use iced::widget::operation::AbsoluteOffset;
use iced::{Element, Subscription, Task, Theme, stream};
use image::DynamicImage;
use serde::Serialize;
use wincode::len::BincodeLen;

pub struct Paths {
    pub config_path: PathBuf,
    pub embedding_cache_path: PathBuf,
}

impl Paths {
    pub fn new_or_die() -> Paths {
        let pd = directories::ProjectDirs::from("", "", "semb2").unwrap();
        Paths {
            config_path: pd.config_dir().to_path_buf().join("config.toml"),
            embedding_cache_path: pd.cache_dir().to_path_buf().join("embeddings.bin"),
        }
    }
}

lazy_static::lazy_static! {
    pub static ref PATHS: Paths = Paths::new_or_die();
}

fn get_pdf_page(file_path: impl AsRef<Path>, page: u32) -> Result<DynamicImage, anyhow::Error> {
    use pdf2image::PDF;
    let pdf = PDF::from_file(file_path.as_ref())?;
    // let page_count = pdf.page_count();
    let mut pages = pdf.render(
        pdf2image::Pages::Range(page..=page),
        pdf2image::RenderOptionsBuilder::default()
            .build()
            .with_context(|| format!("render page {} of {}", page, file_path.as_ref().display()))?,
    )?;
    assert_eq!(pages.len(), 1);
    Ok(pages.drain(..).nth(0).unwrap())
}

fn embed_query(embedder: &ColSmolEmbedder, query: &str) -> Result<Tensor, anyhow::Error> {
    let (input_ids, attention_mask) = embedder
        .processor
        .tokenize_batch(vec![query], &embedder.device)?;

    Ok(embedder
        .model
        .write()
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .forward(&input_ids, &attention_mask, &None, &None)?
        .to_dtype(DType::F32)?)
}

fn normalize(embeddings: Tensor) -> Result<Tensor, anyhow::Error> {
    let ndim = embeddings.rank() - 1;
    let lens = &((&embeddings * &embeddings)?).sum_keepdim((ndim,))?;
    Ok(embeddings.broadcast_div(&lens)?)
}

const PREALLOC_SIZE_LIMIT: usize = 8 * 1024 * 1024 * 1024;
type Max8GB = BincodeLen<PREALLOC_SIZE_LIMIT>;

#[derive(wincode::SchemaRead, wincode::SchemaWrite)]
struct SavedEmbedding {
    #[wincode(with = "wincode::containers::Vec<f32, Max8GB>")]
    data: Vec<f32>,
    shape: Vec<usize>,
    page_sizes: Vec<usize>,
}

impl SavedEmbedding {
    pub fn new(embeddings: PageEmbeddings) -> Self {
        Self {
            data: embeddings.data.flatten_all().unwrap().to_vec1().unwrap(),
            shape: embeddings.data.shape().clone().into_dims(),
            page_sizes: embeddings.page_sizes,
        }
    }

    fn into_loaded(self, device: &candle_core::Device) -> PageEmbeddings {
        let embeddings = Tensor::from_vec(self.data, self.shape, device).unwrap();
        PageEmbeddings {
            data: embeddings,
            page_sizes: self.page_sizes,
        }
    }
}

#[derive(wincode::SchemaRead, wincode::SchemaWrite)]
struct SavedDocData {
    last_mtime: SystemTime,
    embedding: Option<SavedEmbedding>,
}

#[derive(Debug, Clone)]
struct PageEmbeddings {
    /// [of shape n_chunks, embed_dim]
    data: Tensor,
    /// sizes in data [dim 0]. page 0 is data[0..page_sizes[0]], etc
    page_sizes: Vec<usize>,
}

impl PageEmbeddings {
    fn from_pages(pages: Vec<Tensor>) -> Self {
        let sizes = pages.iter().map(|x| x.dim(0).unwrap()).collect();
        let embeddings = Tensor::cat(&pages, 0).expect("have same first dim");
        Self {
            data: embeddings,
            page_sizes: sizes,
        }
    }

    /// compute MaxSim across all pages of a document
    ///
    /// * `query`: query embedding of shape [n_query_chunks, embed_dim]
    fn compute_max_sim(&self, query: &Tensor) -> Result<Vec<f32>, anyhow::Error> {
        const QUERY_DIM: usize = 0;
        const CHUNK_DIM: usize = 1;
        const FEATURE_DIM: usize = 2;

        let qe = query.unsqueeze(CHUNK_DIM)?;
        let de = self.data.unsqueeze(QUERY_DIM)?;
        let dot_product = qe.broadcast_mul(&de)?.sum((FEATURE_DIM,))?;

        let mut start = 0;
        let mut scores = Vec::with_capacity(self.page_sizes.len());
        for p in self.page_sizes.iter() {
            scores.push(
                dot_product
                    .i((.., start..start + p))?
                    .max(CHUNK_DIM)?
                    .sum(QUERY_DIM)?
                    .to_vec0()?,
            );
            start += p;
        }

        Ok(scores)
    }
}

#[derive(Clone, Debug)]
struct DocData {
    last_mtime: SystemTime,
    page_embeddings: Option<PageEmbeddings>,
}

impl Hash for DocData {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.last_mtime.hash(state);
        // self.page_embeddings.hash(state);
    }
}

fn load_model() -> ColSmolEmbedder {
    log::info!("Loading Model");
    let ret = ColSmolEmbedder::new("akshayballal/colSmol-256M-merged", None).unwrap();
    log::info!("Done Loading Model");
    ret
}

#[derive(Clone)]
struct QueryResult {
    path: String,
    page: usize,
    score: f32,
}

struct ConfigEditorState {
    theme: combo_box::State<_Theme>,
    new_seach_path: String,
}

struct State {
    handles: EmbedHandles,
    query: String,
    last_query: String,
    results: Vec<QueryResult>,
    loaded_files: HashMap<String, DocData>,
    init_focus: bool,
    highlight_index: usize,
    config: Config,
    show_config: bool,
    show_log: bool,
    n_docs_total: usize,
    config_editor: ConfigEditorState,
    preview_cache: HashMap<(String, usize), iced::widget::image::Handle>,
    log: Vec<LogMessage>,
}

#[derive(serde::Deserialize, serde::Serialize)]
struct Viewer {
    program_name: String,
    page_arg: String,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
enum _Theme {
    Light,
    Dark,
    Dracula,
    Nord,
    SolarizedLight,
    SolarizedDark,
    GruvboxLight,
    GruvboxDark,
    CatppuccinLatte,
    CatppuccinFrappe,
    CatppuccinMacchiato,
    CatppuccinMocha,
    TokyoNight,
    TokyoNightStorm,
    TokyoNightLight,
    KanagawaWave,
    KanagawaDragon,
    KanagawaLotus,
    Moonfly,
    Nightfly,
    Oxocarbon,
    Ferra,
}

impl std::fmt::Display for _Theme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use _Theme::*;
        let theme = match self {
            Light => "Light",
            Dark => "Dark",
            Dracula => "Dracula",
            Nord => "Nord",
            SolarizedLight => "SolarizedLight",
            SolarizedDark => "SolarizedDark",
            GruvboxLight => "GruvboxLight",
            GruvboxDark => "GruvboxDark",
            CatppuccinLatte => "CatppuccinLatte",
            CatppuccinFrappe => "CatppuccinFrappe",
            CatppuccinMacchiato => "CatppuccinMacchiato",
            CatppuccinMocha => "CatppuccinMocha",
            TokyoNight => "TokyoNight",
            TokyoNightStorm => "TokyoNightStorm",
            TokyoNightLight => "TokyoNightLight",
            KanagawaWave => "KanagawaWave",
            KanagawaDragon => "KanagawaDragon",
            KanagawaLotus => "KanagawaLotus",
            Moonfly => "Moonfly",
            Nightfly => "Nightfly",
            Oxocarbon => "Oxocarbon",
            Ferra => "Ferra",
        };
        f.write_str(theme)
    }
}

impl TryFrom<&Theme> for _Theme {
    type Error = anyhow::Error;

    fn try_from(value: &Theme) -> Result<Self, Self::Error> {
        let t = match value {
            Theme::Light => _Theme::Light,
            Theme::Dark => _Theme::Dark,
            Theme::Dracula => _Theme::Dracula,
            Theme::Nord => _Theme::Nord,
            Theme::SolarizedLight => _Theme::SolarizedLight,
            Theme::SolarizedDark => _Theme::SolarizedDark,
            Theme::GruvboxLight => _Theme::GruvboxLight,
            Theme::GruvboxDark => _Theme::GruvboxDark,
            Theme::CatppuccinLatte => _Theme::CatppuccinLatte,
            Theme::CatppuccinFrappe => _Theme::CatppuccinFrappe,
            Theme::CatppuccinMacchiato => _Theme::CatppuccinMacchiato,
            Theme::CatppuccinMocha => _Theme::CatppuccinMocha,
            Theme::TokyoNight => _Theme::TokyoNight,
            Theme::TokyoNightStorm => _Theme::TokyoNightStorm,
            Theme::TokyoNightLight => _Theme::TokyoNightLight,
            Theme::KanagawaWave => _Theme::KanagawaWave,
            Theme::KanagawaDragon => _Theme::KanagawaDragon,
            Theme::KanagawaLotus => _Theme::KanagawaLotus,
            Theme::Moonfly => _Theme::Moonfly,
            Theme::Nightfly => _Theme::Nightfly,
            Theme::Oxocarbon => _Theme::Oxocarbon,
            Theme::Ferra => _Theme::Ferra,
            Theme::Custom(_) => anyhow::bail!("custom theme cannot be converted to savable theme"),
        };

        Ok(t)
    }
}

impl Into<Theme> for _Theme {
    fn into(self) -> Theme {
        match self {
            _Theme::Light => Theme::Light,
            _Theme::Dark => Theme::Dark,
            _Theme::Dracula => Theme::Dracula,
            _Theme::Nord => Theme::Nord,
            _Theme::SolarizedLight => Theme::SolarizedLight,
            _Theme::SolarizedDark => Theme::SolarizedDark,
            _Theme::GruvboxLight => Theme::GruvboxLight,
            _Theme::GruvboxDark => Theme::GruvboxDark,
            _Theme::CatppuccinLatte => Theme::CatppuccinLatte,
            _Theme::CatppuccinFrappe => Theme::CatppuccinFrappe,
            _Theme::CatppuccinMacchiato => Theme::CatppuccinMacchiato,
            _Theme::CatppuccinMocha => Theme::CatppuccinMocha,
            _Theme::TokyoNight => Theme::TokyoNight,
            _Theme::TokyoNightStorm => Theme::TokyoNightStorm,
            _Theme::TokyoNightLight => Theme::TokyoNightLight,
            _Theme::KanagawaWave => Theme::KanagawaWave,
            _Theme::KanagawaDragon => Theme::KanagawaDragon,
            _Theme::KanagawaLotus => Theme::KanagawaLotus,
            _Theme::Moonfly => Theme::Moonfly,
            _Theme::Nightfly => Theme::Nightfly,
            _Theme::Oxocarbon => Theme::Oxocarbon,
            _Theme::Ferra => Theme::Ferra,
        }
    }
}

fn deserialize_theme_name<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Theme, <D as serde::Deserializer<'de>>::Error> {
    use serde::Deserialize;

    let theme = _Theme::deserialize(d)?;
    Ok(theme.into())
}

fn serialize_theme_name<S: serde::Serializer>(
    t: &Theme,
    s: S,
) -> Result<<S as serde::Serializer>::Ok, S::Error> {
    let t = match t {
        Theme::Light => _Theme::Light,
        Theme::Dark => _Theme::Dark,
        Theme::Dracula => _Theme::Dracula,
        Theme::Nord => _Theme::Nord,
        Theme::SolarizedLight => _Theme::SolarizedLight,
        Theme::SolarizedDark => _Theme::SolarizedDark,
        Theme::GruvboxLight => _Theme::GruvboxLight,
        Theme::GruvboxDark => _Theme::GruvboxDark,
        Theme::CatppuccinLatte => _Theme::CatppuccinLatte,
        Theme::CatppuccinFrappe => _Theme::CatppuccinFrappe,
        Theme::CatppuccinMacchiato => _Theme::CatppuccinMacchiato,
        Theme::CatppuccinMocha => _Theme::CatppuccinMocha,
        Theme::TokyoNight => _Theme::TokyoNight,
        Theme::TokyoNightStorm => _Theme::TokyoNightStorm,
        Theme::TokyoNightLight => _Theme::TokyoNightLight,
        Theme::KanagawaWave => _Theme::KanagawaWave,
        Theme::KanagawaDragon => _Theme::KanagawaDragon,
        Theme::KanagawaLotus => _Theme::KanagawaLotus,
        Theme::Moonfly => _Theme::Moonfly,
        Theme::Nightfly => _Theme::Nightfly,
        Theme::Oxocarbon => _Theme::Oxocarbon,
        Theme::Ferra => _Theme::Ferra,
        Theme::Custom(_) => Err(serde::ser::Error::custom(
            "Serialization of custom themes is not implemented",
        ))?,
    };

    t.serialize(s)
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Config {
    text_height: f32,
    padding: f32,
    spacing: f32,
    viewer: Viewer,
    #[serde(
        deserialize_with = "deserialize_theme_name",
        serialize_with = "serialize_theme_name"
    )]
    theme: Theme,
    root_paths: Vec<PathBuf>,
    show_preview: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            text_height: 20.0,
            padding: 5.0,
            spacing: 5.0,
            viewer: Viewer {
                program_name: "okular".to_string(),
                page_arg: "--page".to_string(),
            },
            theme: Theme::Light,
            root_paths: Vec::new(),
            show_preview: true,
        }
    }
}

#[derive(Clone, Debug)]
enum Message {
    InputChanged(String),
    RunQuery,
    QueryProcessed(String, Tensor),
    DocumentProcessed,
    ToggleLog,
    FocusInput,
    ToggleConfig,
    Open,
    MoveHighlight { down: bool },
    SetTheme(_Theme),
    SetSpacing(f32),
    SetPadding(f32),
    SetTextHeight(f32),
    SetViewerProgram(String),
    SetViewerPageArg(String),
    SetNewSearchPath(String),
    AddRootPath,
    RemoveSearchPath(usize),
    ToggleShowPreview,
    NewLogMessage,
    ChangeNTotal,
}

fn save_config(cfg: &Config) -> Result<(), anyhow::Error> {
    let serialized = toml::to_string_pretty(cfg).context("serialize config")?;
    use std::io::Write;

    let to = &PATHS.config_path;
    std::fs::create_dir_all(to.parent().expect("is file"))?;
    std::fs::File::create(to)
        .map_err(anyhow::Error::new)
        .context("(re-)create config file")
        .map(BufWriter::new)
        .and_then(|mut x| {
            x.write(serialized.as_bytes())
                .context("Write config to file")?;
            Ok(())
        })?;

    Ok(())
}

fn save_embeddings(embeddings: &HashMap<String, DocData>) -> Result<(), anyhow::Error> {
    std::fs::create_dir_all(&PATHS.embedding_cache_path.parent().expect("is file"))?;
    std::fs::File::create(&PATHS.embedding_cache_path)
        .map_err(anyhow::Error::new)
        .map(BufWriter::new)
        .and_then(|mut x| {
            x.write_all(
                &wincode::serialize(
                    &embeddings
                        .iter()
                        .map(
                            |(
                                file,
                                DocData {
                                    last_mtime,
                                    page_embeddings,
                                },
                            )| {
                                (
                                    file,
                                    SavedDocData {
                                        last_mtime: last_mtime.clone(),
                                        embedding: page_embeddings
                                            .as_ref()
                                            .map(|x| SavedEmbedding::new(x.clone())),
                                    },
                                )
                            },
                        )
                        .collect::<HashMap<_, _>>(),
                )
                .context("Serialization")?,
            )
            .context("Write embeddings to file")
        })
}

fn update_preview_cache(state: &mut State, path: String, page_number: usize) {
    if !state
        .preview_cache
        .contains_key(&(path.clone(), page_number))
    {
        match get_pdf_page(&path, page_number as u32) {
            Ok(page) => {
                log::info!(
                    "rendered page {page_number} of {path} with resolution {}x{}",
                    page.width(),
                    page.height()
                );
                state.preview_cache.insert(
                    (path.clone(), page_number),
                    Handle::from_rgba(page.width(), page.height(), page.to_rgba8().to_vec()),
                );
            }
            Err(err) => {
                log::warn!(
                    "Could not render image preview for page {} of {}: {err}",
                    page_number,
                    path
                );
            }
        }
    }
}

fn update(state: &mut State, message: Message) -> Task<Message> {
    match message {
        Message::InputChanged(s) => {
            state.query = s;
            Task::none()
        }
        Message::QueryProcessed(query, query_embedding) => {
            state.last_query = query;

            state.highlight_index = 0;
            state.results.clear();
            for (path, v) in state.loaded_files.iter() {
                let Some(ref doc_embedding) = v.page_embeddings else {
                    continue;
                };

                // TODO: think about moving this back to the GPU if it's too slow
                let scores = doc_embedding.compute_max_sim(&query_embedding).unwrap();

                for (i, score) in scores.into_iter().enumerate() {
                    state.results.push(QueryResult {
                        path: path.to_string(),
                        page: i + 1,
                        score,
                    })
                }
            }

            state
                .results
                .sort_by(|a, b| f32::total_cmp(&b.score, &a.score));

            if state.results.len() > 0 {
                let result = &state.results[state.highlight_index];
                update_preview_cache(state, result.path.clone(), result.page);
            }

            iced::widget::operation::scroll_to(
                "results",
                AbsoluteOffset {
                    x: None,
                    y: Some(0.0),
                },
            )
        }
        Message::RunQuery => {
            if state.query == state.last_query {
                log::info!("Opening After 2x RunQuery");
                Task::done(Message::Open)
            } else {
                state
                    .handles
                    .query_request_tx
                    .send(state.query.clone())
                    .unwrap();

                let h2 = state.handles.query_embedding_rx.clone();
                Task::perform(
                    async move {
                        match h2.recv() {
                            Ok(v) => v,
                            Err(_) => panic!("foo"),
                        }
                    },
                    |x| Message::QueryProcessed(x.0, x.1),
                )
            }
        }
        Message::DocumentProcessed => {
            for (path, doc_data) in state.handles.file_embedding_rx.try_iter() {
                log::info!("Finished processing {path}");
                if state
                    .config
                    .root_paths
                    .iter()
                    .map(|x| x.to_string_lossy().to_string())
                    .any(|x| path.starts_with(&x))
                {
                    state.loaded_files.insert(path, doc_data);
                    save_embeddings(&state.loaded_files).unwrap_or_else(|err| {
                        log::error!(
                            "Could not save embeddings to {}: {err}",
                            PATHS.embedding_cache_path.display()
                        );
                    });
                } else {
                    log::info!(
                        "Skipping adding of file {path} to loaded files. No longer matches a root path"
                    );
                }
            }

            Task::none()
        }
        Message::FocusInput => {
            state.init_focus = true;
            iced::widget::operation::focus("query_widget")
        }
        Message::MoveHighlight { down } => {
            let n_results = state.results.len();
            if n_results > 0 {
                if down {
                    state.highlight_index =
                        (state.highlight_index as i64 + 1).rem_euclid(n_results as i64) as usize;
                } else {
                    state.highlight_index =
                        (state.highlight_index as i64 - 1).rem_euclid(n_results as i64) as usize;
                }

                let result = &state.results[state.highlight_index];
                update_preview_cache(state, result.path.clone(), result.page);
            }
            iced::widget::operation::scroll_to(
                "results",
                AbsoluteOffset {
                    x: None,
                    y: Some(
                        state.highlight_index as f32
                            * (state.config.text_height + state.config.padding),
                    ),
                },
            )
        }
        Message::Open => {
            if state.results.len() > 0 {
                let selection = &state.results[state.highlight_index];
                log::info!(
                    "Opening current selection: {} at page {}",
                    selection.path,
                    selection.page
                );
                match std::process::Command::new(state.config.viewer.program_name.clone())
                    .args(&[
                        selection.path.clone(),
                        state.config.viewer.page_arg.clone(),
                        selection.page.to_string(),
                    ])
                    .spawn()
                {
                    Ok(_) => return iced::exit(),
                    Err(err) => log::error!("Could not spawn viewer: {err}"),
                }
            }

            Task::none()
        }
        Message::ToggleConfig => {
            state.show_config = !state.show_config;
            Task::none()
        }
        Message::SetTheme(_theme) => {
            state.config.theme = _theme.into();
            save_config(&state.config).unwrap();
            Task::none()
        }
        Message::SetSpacing(spacing) => {
            state.config.spacing = spacing;
            save_config(&state.config).unwrap();
            Task::none()
        }
        Message::SetPadding(padding) => {
            state.config.padding = padding;
            save_config(&state.config).unwrap();
            Task::none()
        }
        Message::SetViewerProgram(program) => {
            state.config.viewer.program_name = program;
            save_config(&state.config).unwrap();
            Task::none()
        }
        Message::SetViewerPageArg(arg) => {
            state.config.viewer.page_arg = arg;
            save_config(&state.config).unwrap();
            Task::none()
        }
        Message::RemoveSearchPath(idx) => {
            let root = state.config.root_paths.remove(idx);
            let before = state.loaded_files.len();
            state.loaded_files = state
                .loaded_files
                .drain()
                .filter(|(path, _data)| !PathBuf::from(path).starts_with(&root))
                .collect();
            let after = state.loaded_files.len();
            state.n_docs_total -= before - after;
            Task::none()
        }
        Message::AddRootPath => {
            let mut new_path = String::new();
            std::mem::swap(&mut new_path, &mut state.config_editor.new_seach_path);
            let n = locate_documents(state, &new_path);
            state.n_docs_total += n;
            state.config.root_paths.push(new_path.into());
            save_config(&state.config).unwrap();

            Task::none()
        }
        Message::SetNewSearchPath(path) => {
            state.config_editor.new_seach_path = path;
            Task::none()
        }
        Message::SetTextHeight(th) => {
            state.config.text_height = th;
            save_config(&state.config).unwrap();

            Task::none()
        }
        Message::ToggleShowPreview => {
            state.config.show_preview = !state.config.show_preview;
            save_config(&state.config).unwrap();

            Task::none()
        }
        Message::NewLogMessage => {
            for log_msg in state.handles.log_rx.try_iter() {
                state.log.push(log_msg)
            }

            Task::none()
        }
        Message::ToggleLog => {
            state.show_log = !state.show_log;

            Task::none()
        }
        Message::ChangeNTotal => {
            for diff in state.handles.num_files_rx.try_iter() {
                if diff < 0 && (-diff) as usize > state.n_docs_total {
                    state.n_docs_total = 0;
                } else {
                    state.n_docs_total = (state.n_docs_total as isize + diff) as usize;
                }
            }
            log::info!("updated n_docs_total to {}", state.n_docs_total);

            Task::none()
        }
    }
}

fn results(state: &State) -> Element<'_, Message> {
    use iced::widget::{column, container, rich_text, scrollable, span};
    let palette = state.config.theme.palette();

    container(
        scrollable(
            column(
                state
                    .results
                    .iter()
                    .enumerate()
                    .map(|(i, QueryResult { path, page, .. })| {
                        let fg = if i == state.highlight_index {
                            palette.success
                        } else {
                            palette.text
                        };

                        container(
                            rich_text([span(format!("{path} {page}\n")).color(fg)])
                                .on_link_click(iced::never)
                                .size(state.config.text_height),
                        )
                        .width(Fill)
                        .center_y(Fill)
                        .padding(state.config.padding)
                        .style(container::rounded_box)
                        .id(format!("result-{i}"))
                        .into()
                    })
                    .collect::<Vec<_>>(),
            )
            .spacing(state.config.spacing),
        )
        .id("results"),
    )
    .padding(state.config.padding)
    .style(container::rounded_box)
    .into()
}

fn theme(s: &State) -> iced::Theme {
    s.config.theme.clone()
}

fn centered_text<'a>(
    t: impl iced::widget::text::IntoFragment<'a>,
    padding: f32,
    th: f32,
) -> Element<'a, Message> {
    use iced::widget::{container, text};

    container(text(t).center().size(th).height(2.0 * padding + th)).into()
}

fn config_editor(state: &State) -> iced::Element<'_, Message> {
    use iced::widget::{button, column, combo_box, row, slider, space, text, text_input};
    let th = state.config.text_height;
    let pad = state.config.padding;
    let spc = state.config.spacing;
    column!(
        row!(
            button(text("❌").center().size(th)).on_press(Message::ToggleConfig),
            space().width(Fill),
            centered_text("Settings", pad, th),
            space().width(Fill),
        ),
        row!(
            space().width(spc),
            centered_text("Select Theme", pad, th),
            space().width(spc),
            combo_box(
                &state.config_editor.theme,
                "Enter a theme",
                Some(&(&state.config.theme).try_into().unwrap()),
                Message::SetTheme,
            )
            .size(th),
            space().width(4.0 * spc),
            centered_text("Text Size", pad, th),
            space().width(spc),
            slider(3.0..=50.0, th, Message::SetTextHeight)
        ),
        row!(
            centered_text("Spacing", pad, th),
            space().width(spc),
            slider(1.0..=50.0, spc, Message::SetSpacing),
            space().width(4.0 * spc),
            centered_text("Padding", pad, th),
            space().width(spc),
            slider(1.0..=50.0, pad, Message::SetPadding)
        ),
        row!(
            centered_text("Show Page Preview", pad, th),
            space().width(spc),
            iced::widget::checkbox(state.config.show_preview)
                .on_toggle(|_| Message::ToggleShowPreview),
        ),
        iced::widget::rule::horizontal(spc / 4.0),
        row!(
            centered_text("Viewer Program", pad, th),
            space().width(spc),
            text_input(
                "your favourite pdf viewer",
                &state.config.viewer.program_name
            )
            .size(th)
            .on_input(Message::SetViewerProgram),
            space().width(4.0 * spc),
            centered_text("Viewer Page Argument", pad, th),
            space().width(state.config.spacing),
            text_input(
                "How to pass the page to open to your viewer",
                &state.config.viewer.page_arg
            )
            .size(th)
            .on_input(Message::SetViewerPageArg)
        ),
        iced::widget::rule::horizontal(spc / 4.0),
        row!(
            centered_text("Root Paths", pad, th),
            space().width(state.config.spacing),
            text_input("/path/to/documents/", &state.config_editor.new_seach_path)
                .size(th)
                .on_input(Message::SetNewSearchPath)
                .on_submit(Message::AddRootPath),
        ),
        row!(column(state.config.root_paths.iter().enumerate().map(
            |(i, p)| {
                row!(
                    space().width(spc * 4.0),
                    button(text("❌").size(th).center()).on_press(Message::RemoveSearchPath(i)),
                    space().width(spc),
                    centered_text(p.to_string_lossy(), pad, th),
                )
                .into()
            }
        ))),
        iced::widget::rule::horizontal(spc / 4.0),
    )
    .padding(pad)
    .spacing(spc)
    .into()
}

fn show_log(state: &State) -> iced::Element<'_, Message> {
    use iced::widget::{column, container, scrollable, text};

    container(
        scrollable(column(state.log.iter().rev().map(|x| {
            text(format!("[{}] {}", x.time.format("%H:%M:%S"), x.msg)).into()
        })))
        .height(state.log.len().min(3) as f32 * (state.config.text_height + state.config.padding))
        .width(Fill),
    )
    .padding(state.config.padding)
    .into()
}

fn view(state: &State) -> iced::Element<'_, Message> {
    use iced::widget::{button, column, container, row, space, text, text_input};
    let Config {
        text_height: th,
        padding: pad,
        spacing: spc,
        viewer: _,
        theme: _,
        root_paths: _,
        show_preview: _,
    } = state.config;

    let ti = container(
        text_input("Enter Your Query", &state.query)
            .on_input(Message::InputChanged)
            .on_submit(Message::RunQuery)
            .id("query_widget"),
    )
    .padding(pad);

    let loaded = state.loaded_files.len();
    let total = state.n_docs_total;
    let mut pb = iced::widget::progress_bar(0.0..=total as f32, loaded as f32)
        .girth(state.config.text_height);
    if loaded == total {
        pb = pb.style(iced::widget::progress_bar::success);
    } else {
        pb = pb.style(iced::widget::progress_bar::secondary);
    }

    let mut c: Element<'_, Message> = column!(
        row!(
            button(text("⚙").center().size(th)).on_press(Message::ToggleConfig),
            space().width(spc),
            container(pb)
                .height(2.0 * pad + th)
                .center_y(2.0 * pad + th),
            space().width(state.config.spacing),
            centered_text(format!("Loaded {} / {total} documents", loaded), pad, th)
        ),
        ti,
        container(results(state))
            .width(Fill)
            .height(Fill)
            .padding([state.config.padding, 0.0])
    )
    .padding(state.config.padding)
    .into();

    if state.show_log {
        c = column!(c, space().height(spc / 4.0), show_log(state)).into();
    }

    if state.show_config {
        let editor = config_editor(state);
        c = column!(editor, c).into();
    }

    if state.config.show_preview && state.results.len() > 0 {
        use iced::widget::Image;
        let result = state.results[state.highlight_index].clone();
        let img: Element<'_, Message> = state
            .preview_cache
            .get(&(result.path.clone(), result.page))
            .map(|img| Image::new(img.clone()).expand(true).height(Fill).into())
            .unwrap_or_else(|| {
                text(format!(
                    "could not load image for {} page {}",
                    result.path, result.page,
                ))
                .into()
            });

        c = row!(
            c,
            iced::widget::rule::vertical(state.config.spacing / 4.0),
            img
        )
        .into();
    }

    c
}

struct LogMessage {
    msg: String,
    time: chrono::DateTime<chrono::Local>,
}

struct EmbedThreadHandles {
    query_request_rx: crossbeam_channel::Receiver<String>,
    query_embedding_tx: crossbeam_channel::Sender<(String, Tensor)>,

    file_request_rx: crossbeam_channel::Receiver<PathBuf>,
    file_embedding_tx: crossbeam_channel::Sender<(String, DocData)>,

    num_files_tx: crossbeam_channel::Sender<isize>,

    log_tx: crossbeam_channel::Sender<LogMessage>,
}

fn send_log(
    log_tx: &crossbeam_channel::Sender<LogMessage>,
    msg: String,
) -> Result<(), anyhow::Error> {
    log_tx
        .send(LogMessage {
            msg,
            time: chrono::Local::now(),
        })
        .context("send log message")
}

struct EmbedHandles {
    query_request_tx: crossbeam_channel::Sender<String>,
    query_embedding_rx: crossbeam_channel::Receiver<(String, Tensor)>,

    file_request_tx: crossbeam_channel::Sender<PathBuf>,
    file_embedding_rx: crossbeam_channel::Receiver<(String, DocData)>,

    num_files_rx: crossbeam_channel::Receiver<isize>,

    log_rx: crossbeam_channel::Receiver<LogMessage>,
}

impl EmbedThreadHandles {
    fn create() -> (EmbedThreadHandles, EmbedHandles) {
        let (query_request_tx, query_request_rx) = crossbeam_channel::unbounded();
        let (query_embedding_tx, query_embedding_rx) = crossbeam_channel::unbounded();

        let (file_request_tx, file_request_rx) = crossbeam_channel::unbounded();
        let (file_embedding_tx, file_embedding_rx) = crossbeam_channel::unbounded();

        let (num_files_tx, num_files_rx) = crossbeam_channel::unbounded();

        let (log_tx, log_rx) = crossbeam_channel::unbounded();

        (
            EmbedThreadHandles {
                query_request_rx,
                query_embedding_tx,
                file_request_rx,
                file_embedding_tx,
                log_tx,
                num_files_tx,
            },
            EmbedHandles {
                query_request_tx,
                query_embedding_rx,
                file_request_tx,
                file_embedding_rx,
                log_rx,
                num_files_rx,
            },
        )
    }
}

fn embed_single_page(
    image: DynamicImage,
    embedder: &ColSmolEmbedder,
) -> Result<Tensor, anyhow::Error> {
    let (input_ids, attention_mask, pixel_values, pixel_attention_mask) = embedder
        .processor
        .preprocess(&[image], &embedder.device)
        .context("error in preprocessing")?;

    let image_embeddings = embedder
        .model
        .write()
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .forward(
            &input_ids,
            &attention_mask,
            &Some(pixel_values),
            &pixel_attention_mask,
        )
        .context("Error in running model")?
        .to_dtype(DType::F32)?;

    let image_embeddings = image_embeddings
        .unsqueeze(0)
        .context("squeeze before pooling")?
        .avg_pool2d((2, 1))
        .context("token pooling")?
        .squeeze(0)
        .context("squeeze after pooling")?;

    Ok(image_embeddings)
}

fn embed_worker(
    query_rx: crossbeam_channel::Receiver<String>,
    page_rx: crossbeam_channel::Receiver<(PathBuf, usize, usize, DynamicImage)>,
    query_tx: crossbeam_channel::Sender<(String, Tensor)>,
    page_tx: crossbeam_channel::Sender<(PathBuf, usize, Tensor)>,
    log_tx: crossbeam_channel::Sender<LogMessage>,
) {
    let embedder = load_model();

    loop {
        crossbeam_channel::select_biased! {
            recv(&query_rx) -> res => match res {
                Ok(query) => {
                    log::info!("Received embedding request for query {query}");
                    let _ = send_log(&log_tx, format!("Embedding query: '{query}'"));
                    let embedded = embed_query(&embedder, &query).and_then(normalize).unwrap().squeeze(0).unwrap();
                    let _ = query_tx.send((query, embedded.to_device(&candle_core::Device::Cpu).unwrap())).unwrap();
                }
                Err(e) => {
                    log::error!(
                    "Error in embedding thread trying to receive from query request channel: {e}. Aborting..."
                );
                    break;
                }
            },
            recv(&page_rx) -> res => match res {
                Ok((path, page_no, n_pages, image)) => {
                    let embedding = embed_single_page(image, &embedder).unwrap().squeeze(0).unwrap();
                    let _ = send_log(&log_tx, format!("Embedding Page {}/{} of {}", page_no + 1, n_pages, path.display())).unwrap();
                    let _ = page_tx.send((path, page_no, embedding.to_device(&candle_core::Device::Cpu).unwrap())).unwrap();
                }
                Err(e) => {
                    log::error!("[embed_worker] Could not receive from channel: {e}. Exiting...");
                    break;
                }
            }
        }
    }
}

fn handle_file_request(
    path: impl AsRef<Path>,
    file_embedding_tx: &crossbeam_channel::Sender<(String, DocData)>,
    page_request_tx: &crossbeam_channel::Sender<(PathBuf, usize, usize, DynamicImage)>,
    current_documents: &mut HashMap<PathBuf, (Vec<Option<Tensor>>, SystemTime)>,
) -> bool {
    let meta = std::fs::metadata(&path).unwrap();
    let modified = meta.modified().unwrap();

    let mut pages = match embed_anything::embeddings::local::colpali::get_images_from_pdf(&path) {
        Ok(pages) => pages,
        Err(err) => {
            log::error!(
                "Could not render pages of document {}: {err}. Skipping...",
                path.as_ref().display()
            );
            let doc_data = DocData {
                last_mtime: modified,
                page_embeddings: None,
            };
            file_embedding_tx
                .send((path.as_ref().to_string_lossy().to_string(), doc_data))
                .unwrap();

            return false;
        }
    };

    let mut page_data = Vec::<Option<Tensor>>::new();
    let n_pages = pages.len();
    page_data.resize(n_pages, None);
    current_documents.insert(path.as_ref().to_path_buf(), (page_data, modified));

    for (i, p) in pages.drain(..).enumerate() {
        log::info!(
            "Sending page embed request for page {}/{n_pages} of {}",
            i,
            path.as_ref().display()
        );
        page_request_tx
            .send((path.as_ref().to_path_buf(), i, n_pages, p))
            .unwrap();
    }

    true
}

fn embed_thread(et: EmbedThreadHandles, root_paths: Vec<impl AsRef<Path>>) {
    let mut current_documents: HashMap<PathBuf, (Vec<Option<Tensor>>, SystemTime)> = HashMap::new();
    let (query_request_tx, query_request_rx) = crossbeam_channel::unbounded();
    let (page_request_tx, page_request_rx) = crossbeam_channel::unbounded();
    let (query_embedding_tx, query_embedding_rx) = crossbeam_channel::unbounded();
    let (page_embedding_tx, page_embedding_rx) = crossbeam_channel::unbounded();
    let (own_file_request_tx, own_file_request_rx) = crossbeam_channel::unbounded::<PathBuf>();

    let lt = et.log_tx.clone();
    let pt = page_embedding_tx.clone();
    let handle = std::thread::spawn(|| {
        embed_worker(
            query_request_rx,
            page_request_rx,
            query_embedding_tx,
            pt,
            lt,
        )
    });

    {
        let mut loaded_files = load_saved_embeddings(
            &std::path::PathBuf::from(&PATHS.embedding_cache_path),
            &candle_core::Device::Cpu,
        );

        for p in root_paths {
            for f in walkdir::WalkDir::new(p).into_iter().filter_map(|f| {
                match f {
                    Ok(f)
                        if f.file_name()
                            .to_str()
                            .expect("should be convertible to str")
                            .ends_with(".pdf") =>
                    {
                        Some(f.into_path())
                    }
                    Ok(_) => {
                        // not a pdf
                        None
                    }
                    Err(e) => {
                        log::warn!("Could not load file: {e}");
                        None
                    }
                }
            }) {
                let meta = std::fs::metadata(&f).unwrap();
                let modified = meta.modified().unwrap();
                let key = f.to_string_lossy().to_string();

                if loaded_files.contains_key(&key) && loaded_files[&key].last_mtime >= modified {
                    log::info!("Skipping {key}. Already embedded before");
                } else {
                    log::info!("Queuing embedding of {key}");
                    own_file_request_tx.send(f).unwrap();
                    et.num_files_tx.send(1).unwrap();
                }
            }
        }

        et.num_files_tx.send(loaded_files.len() as isize).unwrap();

        for (file, embed) in loaded_files.drain() {
            et.file_embedding_tx.send((file, embed)).unwrap();
        }
    }

    loop {
        log::info!(
            "Current page embeddings queue size: {}",
            &page_request_tx.len()
        );

        let mut select = crossbeam_channel::Select::new_biased();
        select.recv(&query_embedding_rx);
        select.recv(&et.query_request_rx);
        select.recv(&page_embedding_rx);

        let file_request_receivers = if page_request_tx.len() < 500 {
            select.recv(&own_file_request_rx);
            select.recv(&et.file_request_rx);
            Some([&own_file_request_rx, &et.file_request_rx])
        } else {
            None
        };

        let op = select.select();

        match op.index() {
            0 => match op.recv(&query_embedding_rx) {
                Ok(qe) => et.query_embedding_tx.send(qe).unwrap(),
                Err(err) => {
                    log::error!(
                        "[embed_thread] could not receive query embedding: {err}. Aborting"
                    );
                    break;
                }
            },
            1 => match op.recv(&et.query_request_rx) {
                Ok(q) => query_request_tx.send(q).unwrap(),
                Err(err) => {
                    log::error!(
                        "[embed_thread] could not receive query embedding request: {err}. Aborting"
                    );
                    break;
                }
            },
            2 => match op.recv(&page_embedding_rx) {
                Ok((path, page_no, embedding)) => {
                    let finished = {
                        let Some((embeddings, _)) = current_documents.get_mut(&path) else {
                            et.num_files_tx.send(-1).unwrap();
                            log::warn!(
                                "[embed_thread] Got embedding for document which has been canceled ({}). Skipping.",
                                path.display()
                            );
                            break;
                        };
                        embeddings[page_no] = Some(embedding);
                        embeddings.iter().all(|x| x.is_some())
                    };
                    if finished {
                        let (mut embeddings, modified) = current_documents
                            .remove(&path)
                            .expect("we use it just above");
                        let embeddings = embeddings
                            .drain(..)
                            .map(|x| x.expect("we check all are Some above"))
                            .collect::<Vec<_>>();
                        let doc_data = DocData {
                            last_mtime: modified,
                            page_embeddings: Some(PageEmbeddings::from_pages(embeddings)),
                        };
                        et.file_embedding_tx
                            .send((path.to_string_lossy().to_string(), doc_data))
                            .unwrap();
                    }
                }
                Err(err) => {
                    log::error!(
                        "[embed_thread] could not receive page embedding request: {err}. Aborting"
                    );
                    break;
                }
            },
            x if x == 3 || x == 4 => {
                let file_request_receivers = file_request_receivers
                    .expect("this branch can only happen if we have file requests in select");
                match op.recv(file_request_receivers[x - 3]) {
                    Ok(path) => {
                        log::info!("Received embedding request for document {}", path.display());
                        if !handle_file_request(
                            &path,
                            &et.file_embedding_tx,
                            &page_request_tx,
                            &mut current_documents,
                        ) {
                            send_log(
                                &et.log_tx,
                                format!(
                                    "Skipped file {} due to error in page processing.",
                                    path.display()
                                ),
                            )
                            .unwrap();
                        };
                    }
                    Err(e) => {
                        log::error!(
                            "Error in embedding thread trying to receive from file request channel: {e}. Aborting"
                        );
                        break;
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    let _ = handle.join();
    log::info!("Embed thread finished")
}

fn new() -> State {
    let config: Config = std::fs::File::open(&PATHS.config_path)
        .map_err(|x| anyhow::Error::new(x).context("open config file"))
        .map(BufReader::new)
        .and_then(|mut x| {
            let mut buf = String::new();
            x.read_to_string(&mut buf).context("read config file")?;
            Ok(buf)
        })
        .and_then(|x| toml::from_str(x.as_ref()).context("parse configuration"))
        .unwrap_or_default();
    let (et, handles) = EmbedThreadHandles::create();

    let rp = config.root_paths.clone();
    std::thread::spawn(|| embed_thread(et, rp));

    let state = State {
        query: String::new(),
        results: Vec::new(),
        loaded_files: HashMap::new(),
        handles,
        init_focus: false,
        highlight_index: 0,
        config,
        last_query: String::from(""),
        show_config: false,
        config_editor: ConfigEditorState {
            theme: combo_box::State::new(vec![
                _Theme::Light,
                _Theme::Dark,
                _Theme::Dracula,
                _Theme::Nord,
                _Theme::SolarizedLight,
                _Theme::SolarizedDark,
                _Theme::GruvboxLight,
                _Theme::GruvboxDark,
                _Theme::CatppuccinLatte,
                _Theme::CatppuccinFrappe,
                _Theme::CatppuccinMacchiato,
                _Theme::CatppuccinMocha,
                _Theme::TokyoNight,
                _Theme::TokyoNightStorm,
                _Theme::TokyoNightLight,
                _Theme::KanagawaWave,
                _Theme::KanagawaDragon,
                _Theme::KanagawaLotus,
                _Theme::Moonfly,
                _Theme::Nightfly,
                _Theme::Oxocarbon,
                _Theme::Ferra,
            ]),
            new_seach_path: "".to_string(),
        },
        preview_cache: HashMap::new(),
        show_log: false,
        log: Vec::new(),
        n_docs_total: 0,
    };

    state
}

fn locate_documents(state: &mut State, root: impl AsRef<Path>) -> usize {
    let mut n = 0;
    for f in walkdir::WalkDir::new(root).into_iter().filter_map(|f| {
        match f {
            Ok(f)
                if f.file_name()
                    .to_str()
                    .expect("should be convertible to str")
                    .ends_with(".pdf") =>
            {
                Some(f.into_path())
            }
            Ok(_) => {
                // not a pdf
                None
            }
            Err(e) => {
                log::warn!("Could not load file: {e}");
                None
            }
        }
    }) {
        let meta = std::fs::metadata(&f).unwrap();
        let modified = meta.modified().unwrap();
        let key = f.to_string_lossy().to_string();

        if state.loaded_files.contains_key(&key) && state.loaded_files[&key].last_mtime >= modified
        {
            log::info!("Skipping {key}. Already embedded before");
        } else {
            log::info!("Queuing embedding of {key}");
            state.handles.file_request_tx.send(f).unwrap();
            n += 1;
        }
    }
    n
}

fn load_saved_embeddings(
    save_path: &impl AsRef<std::path::Path>,
    device: &candle_core::Device,
) -> HashMap<String, DocData> {
    std::fs::File::open(save_path)
        .map_err(|x| anyhow::Error::from(x))
        .map(std::io::BufReader::new)
        .and_then(|mut x| {
            {
                let mut buf = Vec::new();
                x.read_to_end(&mut buf).context("read file")?;
                Ok(buf)
            }
            .and_then(|x| {
                wincode::deserialize::<HashMap<String, SavedDocData>>(&x)
                    .context("Error in deserialization")
            })
        })
        .map(|v| {
            v.into_iter()
                .map(|(key, value)| {
                    (
                        key,
                        DocData {
                            last_mtime: value.last_mtime,
                            page_embeddings: value.embedding.map(|x| x.into_loaded(device)),
                        },
                    )
                })
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_else(|e| {
            log::info!("Could not properly deserialize embeddings: {e}. starting from scratch");
            HashMap::new()
        })
}

fn main() -> iced::Result {
    colog::init();
    iced::application(new, update, view)
        .subscription(|state: &State| {
            let mut subscriptions = Vec::new();
            if !state.init_focus {
                subscriptions.push(
                    iced::time::every(std::time::Duration::from_millis(250))
                        .map(|_| Message::FocusInput),
                );
            }

            if state.handles.num_files_rx.len() > 0 {
                subscriptions.push(
                    iced::time::every(std::time::Duration::from_millis(250))
                        .map(|_| Message::ChangeNTotal),
                );
            }

            if state.handles.file_embedding_rx.len() > 0 {
                subscriptions.push(
                    iced::time::every(std::time::Duration::from_millis(250))
                        .map(|_| Message::DocumentProcessed),
                );
            }

            if state.show_log {
                subscriptions.push(
                    iced::time::every(std::time::Duration::from_millis(250))
                        .map(|_| Message::NewLogMessage),
                );
            }

            subscriptions.push(iced::keyboard::listen().filter_map(|x| match x {
                iced::keyboard::Event::KeyPressed {
                    key,
                    modified_key: _,
                    physical_key: _,
                    location: _,
                    modifiers,
                    text: _,
                    repeat: _,
                } => {
                    if key == Key::Named(iced::keyboard::key::Named::Tab) {
                        Some(Message::MoveHighlight {
                            down: !modifiers.shift(),
                        })
                    } else if key == Key::Character("o".into()) && modifiers.control() {
                        Some(Message::Open)
                    } else if key == Key::Character("l".into()) && modifiers.control() {
                        Some(Message::FocusInput)
                    } else if key == Key::Character("m".into()) && modifiers.control() {
                        Some(Message::ToggleLog)
                    } else {
                        None
                    }
                }
                iced::keyboard::Event::KeyReleased {
                    key: _,
                    modified_key: _,
                    physical_key: _,
                    location: _,
                    modifiers: _,
                } => None,
                iced::keyboard::Event::ModifiersChanged(_) => None,
            }));
            Subscription::batch(subscriptions)
        })
        .theme(theme)
        .centered()
        .window(iced::window::Settings {
            size: (1200f32, 600f32).into(),
            maximized: false,
            fullscreen: false,
            position: iced::window::Position::Centered,
            min_size: Some((800f32, 600f32).into()),
            max_size: None,
            visible: true,
            resizable: true,
            closeable: true,
            minimizable: false,
            decorations: true,
            transparent: false,
            blur: false,
            level: iced::window::Level::AlwaysOnTop,
            ..Default::default()
        })
        .title("semb2")
        .run()
}
