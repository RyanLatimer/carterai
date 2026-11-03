fn main() {
    println!("cargo:rerun-if-changed=native/proxy.c");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("native/proxy.c")
            .flag("-std=c11")
            .compile("carter_proxy");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=CFNetwork");
    }
}
