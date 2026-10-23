# semb2reamde

local embedding search powered py ColPali / ColSmol

install using
```commandline
$ cargo install --path .
```

## Roadmap
- [x] add previewing of pages when tabbing through results
- [x] improve asynchronous model loading
- [x] allow partial document saves
- [ ] separately load embeddings and metadata. this should allow better asynchronous loading with even lower initial waiting times
- [ ] make embeddings loadable per-file somehow to improve async loading even more
- [ ] Robustness: fix all uses of unwrap
- [ ] dampen dependency hell a bit
- [ ] make double query to open work again
- [x] add status messages
- [x] add placeholders for files which errored during loading so we don't have to load them again
- [ ] regularily update progress bar instead of only on-focus
- [x] log is only shown if results are present
- [x] limiting memory consumption causes embedding thread to block on trying to send to queue. fix
- [ ] improve search performance using HNSWlib
- [ ] when embedding queue is empty, papers aren't being pushed into it
