# semb2reamde

local embedding search powered py ColPali / ColSmol

install using
```commandline
$ cargo install --path .
```

## Roadmap
- [ ] separately load embeddings and metadata. this should allow better asynchronous loading with even lower initial waiting times
- [ ] make embeddings loadable per-file somehow to improve async loading even more
- [ ] Robustness: fix all uses of unwrap
- [ ] dampen dependency hell a bit
- [ ] make double query to open work again
    - problem is that after an enter, the input loses keyboard focus
    - maybe just query whether the enter key was pressed?
- [ ] regularily update progress bar instead of only on-focus
- [ ] implement candidate selection similar to the idea in the PLAID engine (probably similar)
- [ ] quantization of document tokens and / or k-means pooling
- [x] add previewing of pages when tabbing through results
- [x] improve asynchronous model loading
- [x] allow partial document saves
- [x] add status messages
- [x] add placeholders for files which errored during loading so we don't have to load them again
- [x] log is only shown if results are present
- [x] limiting memory consumption causes embedding thread to block on trying to send to queue. fix
- [x] improve search performance using HNSWlib -> impossible with hnsw, because MaxSim is asymmetric
- [x] when embedding queue is empty, papers aren't being pushed into it
- [x] use custom candle cuda kernel for max_sim operator
