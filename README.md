# nnn

(educational) nano neural nets

* Autograd for `f64`

## Running examples

Install [Graphviz](https://graphviz.org/download/) and make sure the `dot` executable is added to the path.

Install Jupyter notebook.

* on Debian or Ubuntu: `sudo apt install jupyter-notebook`
* on macOS: `brew install jupyter`

```sh
# install evcxr_jupyter
cargo install evcxr_jupyter
# add Rust kernel
evcxr_jupyter --install
# if you haven't added the source of Rust std
rustup component add rust-src
# run Jupyter
jupyter notebook
```

## License

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
