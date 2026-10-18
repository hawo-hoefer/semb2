use std::collections::HashMap;
use std::hash::Hash;
use std::io::{BufWriter, Read};
use std::path::PathBuf;
use std::time::SystemTime;

use anyhow::Context;
use candle_core::{DType, Tensor};
use embed_anything::embeddings::local::colsmol::ColSmolEmbedder;
use iced::Length::Fill;
use iced::keyboard::Key;
use iced::widget::operation::AbsoluteOffset;
use iced::{Element, Subscription, Task, stream};
use wincode::len::BincodeLen;

fn max_sim(query_embedding: &Tensor, document_embedding: &Tensor) -> Result<Tensor, anyhow::Error> {
    const DOC_DIM: usize = 2;
    const QUERY_DIM: usize = 1;
    const FEATURE_DIM: usize = 3;
    let qe = query_embedding.unsqueeze(DOC_DIM)?;
    let de = document_embedding.unsqueeze(QUERY_DIM)?;

    let dot_product = qe.broadcast_mul(&de)?.sum((FEATURE_DIM,))?;
    let scores = (dot_product.max(DOC_DIM)?).sum(QUERY_DIM)?;
    Ok(scores)
}

fn embed_file(
    embedder: &ColSmolEmbedder,
    file_path: PathBuf,
    batch_size: usize,
) -> Result<Tensor, anyhow::Error> {
    let pages = embed_anything::embeddings::local::colpali::get_images_from_pdf(&file_path)?;
    let mut embed_data = Vec::new();
    for batch in pages.chunks(batch_size) {
        // let start_page = index * batch_size + 1;
        // let end_page = start_page + batch.len();
        // let page_numbers = (start_page..=end_page).collect::<Vec<_>>();
        let (input_ids, attention_mask, pixel_values, pixel_attention_mask) =
            embedder.processor.preprocess(batch, &embedder.device)?;

        let image_embeddings = embedder
            .model
            .write()
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .forward(
                &input_ids,
                &attention_mask,
                &Some(pixel_values),
                &pixel_attention_mask,
            )?
            .to_dtype(DType::F32)?;
        embed_data.push(image_embeddings);
    }

    Ok(Tensor::cat(&embed_data, 0)?)
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
struct SavedDocData {
    last_mtime: SystemTime,
    #[wincode(with = "wincode::containers::Vec<f32, Max8GB>")]
    page_embeddings: Vec<f32>,
    page_embedding_shape: Vec<usize>,
}

#[derive(Clone, Debug)]
struct DocData {
    last_mtime: SystemTime,
    page_embeddings: Tensor,
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

struct QueryResult {
    path: String,
    page: usize,
    score: f32,
}

enum EmbedRequest {
    File(PathBuf),
    Query(String),
}

#[derive(Clone)]
enum EmbedResult {
    File(String, DocData),
    Query(Tensor),
}

struct State {
    handles: EmbedHandles,
    query: String,
    last_query: String,
    results: Vec<QueryResult>,
    loaded_files: HashMap<String, DocData>,
    init_focus: bool,
    device: candle_core::Device,
    highlight_index: usize,
    theme: iced::theme::Theme,
    config: Config,
}

struct Viewer {
    program_name: String,
    page_arg: String,
}

struct Config {
    text_height: f32,
    padding: f32,
    spacing: f32,
    viewer: Viewer,
}

#[derive(Clone, Debug)]
enum Message {
    InputChanged(String),
    RunQuery,
    QueryProcessed(String, Tensor),
    DocumentProcessed(String, DocData),
    InitFocus,
    Open,
    MoveHighlight { down: bool },
}

fn update(state: &mut State, message: Message) -> Task<Message> {
    match message {
        Message::InputChanged(s) => {
            state.query = s;
            Task::none()
        }
        Message::QueryProcessed(query, embedding) => {
            state.highlight_index = 0;
            state.results.clear();
            for (path, v) in state.loaded_files.iter() {
                let scores = max_sim(
                    &embedding.to_device(&state.device).unwrap(),
                    &v.page_embeddings.to_device(&state.device).unwrap(),
                )
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
                for (i, score) in scores.into_iter().enumerate() {
                    state.results.push(QueryResult {
                        path: path.to_string(),
                        page: i + 1,
                        score,
                    })
                }
            }
            state.last_query = query;
            state
                .results
                .sort_by(|a, b| f32::total_cmp(&b.score, &a.score));

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
        Message::DocumentProcessed(path, doc_data) => {
            log::info!("Finished processing {path}");
            state.loaded_files.insert(path, doc_data);
            let save_path = std::path::PathBuf::from("./embeddings.bin");
            use std::io::Write;

            std::fs::File::create(save_path)
                .map_err(anyhow::Error::new)
                .map(BufWriter::new)
                .and_then(|mut x| {
                    x.write_all(
                        &wincode::serialize(
                            &state
                                .loaded_files
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
                                                page_embedding_shape: page_embeddings
                                                    .shape()
                                                    .clone()
                                                    .into_dims(),
                                                page_embeddings: page_embeddings
                                                    .flatten_all()
                                                    .unwrap()
                                                    .to_vec1()
                                                    .unwrap(),
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
                .unwrap();

            Task::none()
        }
        Message::InitFocus => {
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
            }
            iced::widget::operation::scroll_to(
                "results",
                AbsoluteOffset {
                    x: None,
                    y: Some(
                        state.highlight_index as f32
                            * (state.config.text_height
                                + state.config.padding
                                + state.config.spacing),
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
                    Ok(ok) => (),
                    Err(err) => log::error!("Could not spawn viewer: {err}"),
                }
            }

            Task::none()
        }
    }
}

fn results(state: &State) -> Element<'_, Message> {
    use iced::color;
    use iced::widget::{column, container, rich_text, row, scrollable, span, text, text_input};
    let palette = state.theme.palette();

    container(
        scrollable(column(
            state
                .results
                .iter()
                .enumerate()
                .map(|(i, QueryResult { path, page, score })| {
                    let fg = if i == state.highlight_index {
                        palette.success
                    } else {
                        palette.text
                    };

                    container(
                        rich_text([span(format!("{score:.3} {path} | {page}\n")).color(fg)])
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
        ))
        .id("results"),
    )
    .padding(state.config.padding)
    .style(container::rounded_box)
    .into()
}

fn theme(s: &State) -> iced::Theme {
    s.theme.clone()
}

fn view(state: &State) -> iced::Element<'_, Message> {
    use iced::widget::{column, container, text_input};
    let ti = text_input("Enter Your Query", &state.query)
        .on_input(Message::InputChanged)
        .on_submit(Message::RunQuery)
        .id("query_widget");

    let in_queue = state.handles.file_request_tx.len();
    let loaded = state.loaded_files.len();
    let total = in_queue + loaded;

    container(
        column!(
            container(iced::widget::progress_bar(0.0..=total as f32, loaded as f32).girth(10))
                .padding([state.config.padding, 0.0]),
            ti,
            container(results(state))
                .width(Fill)
                .height(Fill)
                .padding([state.config.padding, 0.0])
        )
        .spacing(state.config.spacing),
    )
    .padding(state.config.padding)
    .into()
}

struct EmbedThread {
    model: ColSmolEmbedder,

    query_request_rx: crossbeam_channel::Receiver<String>,
    query_embedding_tx: crossbeam_channel::Sender<(String, Tensor)>,

    file_request_rx: crossbeam_channel::Receiver<PathBuf>,
    file_embedding_tx: crossbeam_channel::Sender<(String, DocData)>,
}

struct EmbedHandles {
    query_request_tx: crossbeam_channel::Sender<String>,
    query_embedding_rx: crossbeam_channel::Receiver<(String, Tensor)>,

    file_request_tx: crossbeam_channel::Sender<PathBuf>,
    file_embedding_rx: crossbeam_channel::Receiver<(String, DocData)>,
}

impl EmbedThread {
    fn create(model: ColSmolEmbedder) -> (EmbedThread, EmbedHandles) {
        let (query_request_tx, query_request_rx) = crossbeam_channel::unbounded();
        let (query_embedding_tx, query_embedding_rx) = crossbeam_channel::unbounded();

        let (file_request_tx, file_request_rx) = crossbeam_channel::unbounded();
        let (file_embedding_tx, file_embedding_rx) = crossbeam_channel::unbounded();

        (
            EmbedThread {
                model,
                query_request_rx,
                query_embedding_tx,
                file_request_rx,
                file_embedding_tx,
            },
            EmbedHandles {
                query_request_tx,
                query_embedding_rx,
                file_request_tx,
                file_embedding_rx,
            },
        )
    }
}

fn embed_thread(et: EmbedThread) {
    let mut s = crossbeam_channel::Select::new_biased();
    s.recv(&et.query_request_rx);
    s.recv(&et.file_request_rx);

    loop {
        crossbeam_channel::select_biased! {
            recv(et.query_request_rx) -> query => {
                let query = match query {
                    Ok(q) => q,
                    Err(e) => {
                        log::error!(
                            "Error in embedding thread trying to receive from query request channel: {e}. Aborting..."
                        );
                        break;
                    }
                };
                log::info!("Received embedding request for query {query}");
                let embedded = embed_query(&et.model, &query).and_then(normalize).unwrap();
                et.query_embedding_tx.send((query, embedded)).unwrap();
            }
            recv(et.file_request_rx) -> path => {
                let path = match path {
                    Ok(p) => p,
                    Err(e) => {
                        log::error!(
                            "Error in embedding thread trying to receive from file request channel: {e}. Aborting..."
                        );
                        break;
                    }
                };
                log::info!(
                    "Received embedding request for document {path}",
                    path = path.display()
                );
                let meta = std::fs::metadata(&path).unwrap();
                let modified = meta.modified().unwrap();

                let embedded = embed_file(&et.model, path.clone(), 3)
                    .and_then(normalize)
                    .unwrap();
                et.file_embedding_tx.send((path.to_string_lossy().to_string(), DocData { last_mtime: modified, page_embeddings: embedded })).unwrap();
            }
        }
    }
    log::info!("Embed thread finished")
}

fn new() -> State {
    let embed_model = load_model();

    let device = embed_model.device.clone();
    let loaded_files =
        load_saved_embeddings(&std::path::PathBuf::from("./embeddings.bin"), &device);
    let (et, handles) = EmbedThread::create(embed_model);
    std::thread::spawn(|| embed_thread(et));

    for f in walkdir::WalkDir::new("/home/hawo/notes")
        .into_iter()
        .filter_map(|f| {
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
        })
    {
        let meta = std::fs::metadata(&f).unwrap();
        let modified = meta.modified().unwrap();
        let key = f.to_string_lossy().to_string();

        if loaded_files.contains_key(&key) && loaded_files[&key].last_mtime >= modified {
            log::info!("Skipping {key}. Already embedded before");
        } else {
            log::info!("Queuing embedding of {key}");
            handles.file_request_tx.send(f).unwrap();
        }
    }

    State {
        query: String::new(),
        results: Vec::new(),
        loaded_files: loaded_files,
        handles,
        init_focus: false,
        device,
        highlight_index: 0,
        theme: iced::theme::Theme::GruvboxDark,
        config: Config {
            text_height: 20.0,
            padding: 5.0,
            viewer: Viewer {
                program_name: String::from("zathura"),
                page_arg: String::from("--page"),
            },
            spacing: 1.0,
        },
        last_query: String::from(""),
    }
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
                            page_embeddings: Tensor::from_vec(
                                value.page_embeddings,
                                value.page_embedding_shape,
                                device,
                            )
                            .unwrap(),
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

fn init_focus() -> impl iced::futures::Stream<Item = Message> {
    use iced::futures::SinkExt;
    stream::channel(1, async |mut output| {
        output.send(Message::InitFocus).await;
    })
}

fn main() -> iced::Result {
    colog::init();
    iced::application(new, update, view)
        .subscription(|state: &State| {
            let mut subscriptions = Vec::new();
            if !state.init_focus {
                subscriptions.push(Subscription::run(init_focus))
            }
            if state.handles.file_embedding_rx.len() > 0 {
                let embedded = state.handles.file_embedding_rx.recv().unwrap();
                subscriptions.push(Subscription::run_with(embedded, |x| {
                    let path = x.0.clone();
                    let data = x.1.clone();
                    stream::channel(1, async |mut output| {
                        use iced::futures::SinkExt;
                        output
                            .send(Message::DocumentProcessed(path, data))
                            .await
                            .unwrap();
                    })
                }))
            }
            subscriptions.push(iced::keyboard::listen().filter_map(|x| match x {
                iced::keyboard::Event::KeyPressed {
                    key,
                    modified_key: _,
                    physical_key: _,
                    location: _,
                    modifiers,
                    text,
                    repeat: _,
                } => {
                    dbg!(key == Key::Character("o".into()), modifiers.control());
                    if key == Key::Named(iced::keyboard::key::Named::Tab) {
                        Some(Message::MoveHighlight {
                            down: !modifiers.shift(),
                        })
                    } else if key == Key::Character("o".into()) && modifiers.control() {
                        Some(Message::Open)
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
        .run()
}
