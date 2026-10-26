
fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    let bindings = cudaforge::KernelBuilder::new()
        .source_dir("./src/cuda_kernels")
        .build_ptx()
        .expect("failed to build ptx");
    bindings.write("src/cuda_kernels.rs").unwrap();
}
