use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::time::SystemTime;

use anyhow::Context;
use candle_core::backend::BackendDevice;
use candle_core::{CudaDevice, DType, Tensor};
use embed_anything::embeddings::local::colsmol::ColSmolEmbedder;
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
    for (index, batch) in pages.chunks(batch_size).enumerate() {
        let start_page = index * batch_size + 1;
        let end_page = start_page + batch.len();
        let page_numbers = (start_page..=end_page).collect::<Vec<_>>();
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

struct DocData {
    last_mtime: SystemTime,
    page_embeddings: Tensor,
}

fn load_model() -> ColSmolEmbedder {
    log::info!("Loading Model");
    let ret = ColSmolEmbedder::new("akshayballal/colSmol-256M-merged", None).unwrap();
    log::info!("Done Loading Model");
    ret
}

fn main() {
    colog::init();

    let load_model_handle = std::thread::spawn(load_model);

    let path = std::path::Path::new("/home/hawo/notes");
    let save_path = std::path::Path::new("./embeddings.bin");

    let mut embeddings = std::fs::File::open(save_path)
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
                                &candle_core::Device::Cuda(CudaDevice::new(0).unwrap()),
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
        });

    let embed_model = load_model_handle.join().unwrap();

    let query_embedding = embed_query(
        &embed_model,
        "where can i find crystal structure definitions?",
    )
    .and_then(normalize)
    .unwrap();
    for f in walkdir::WalkDir::new(path).into_iter().filter_map(|f| {
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

        if embeddings.contains_key(&key) && embeddings[&key].last_mtime >= modified {
            log::info!("Skipping {key}. Already embedded before")
        } else {
            let embedded = embed_file(&embed_model, f.clone(), 3)
                .and_then(normalize)
                .unwrap();
            log::info!("embedded {}", f.display());
            embeddings.insert(
                f.to_string_lossy().to_string(),
                DocData {
                    last_mtime: modified,
                    page_embeddings: embedded,
                },
            );
        }
    }
    let to_save = embeddings
        .iter()
        .map(|(key, value)| {
            let shape = value.page_embeddings.shape().clone().into_dims();
            (
                key,
                SavedDocData {
                    last_mtime: value.last_mtime,
                    page_embeddings: value
                        .page_embeddings
                        .flatten_all()
                        .unwrap()
                        .to_vec1()
                        .unwrap(),
                    page_embedding_shape: shape,
                },
            )
        })
        .collect::<HashMap<_, _>>();

    std::fs::File::create(save_path)
        .map_err(|x| anyhow::Error::from(x))
        .map(std::io::BufWriter::new)
        .and_then(|x| Ok(wincode::serialize_into(x, &to_save)?))
        .unwrap();
    log::info!("saved to {save_path}", save_path = save_path.display());

    let mut all_scores = Vec::new();
    for (path, v) in embeddings.iter() {
        let scores = max_sim(&query_embedding, &v.page_embeddings)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, score) in scores.into_iter().enumerate() {
            all_scores.push((path, i + 1, score))
        }
    }
    all_scores.sort_by(|(_, _, a), (_, _, b)| f32::total_cmp(a, b));

    for (path, page, score) in all_scores.iter() {
        println!("{score:.4} {path} {page}")
    }
}
