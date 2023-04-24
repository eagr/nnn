use crate::Float64;
use base64::Engine;
use petgraph::graph::{Graph, NodeIndex};
use std::collections::HashMap;
use std::fmt::Display;
use std::io::Write;
use std::process::{Command, Stdio};

pub fn to_graph(root: &Float64) -> Graph<String, &str> {
    let mut graph = Graph::<String, &str>::new();
    let mut indices = HashMap::<Float64, NodeIndex<u32>>::new();

    build_graph(root, &mut graph, &mut indices);
    graph
}

fn build_graph(
    root: &Float64,
    graph: &mut Graph<String, &str>,
    indices: &mut HashMap<Float64, NodeIndex<u32>>,
) -> NodeIndex<u32> {
    if indices.contains_key(root) {
        indices[root]
    } else {
        let inner = root.borrow();
        let op = inner.op;

        let root_idx = graph.add_node(format!("{} | {}", inner.v, inner.g));
        indices.insert(root.clone(), root_idx);

        if op != "" {
            let op_idx = graph.add_node(op.to_string());
            graph.add_edge(op_idx, root_idx, "");

            for child in (&inner.children).iter() {
                let child_idx = build_graph(child, graph, indices);
                graph.add_edge(child_idx, op_idx, "");
            }
        }

        root_idx
    }
}

pub fn draw_dot<T>(dot: T, opts: &[&str]) -> String
where
    T: Display,
{
    // `dot <OPTIONS> <STDIN>`
    let graphviz_proc = Command::new("dot")
        .args(opts)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to run `dot`. Have you installed graphviz?");

    graphviz_proc
        .stdin
        .as_ref()
        .unwrap()
        .write_fmt(format_args!("{}", dot))
        .unwrap();

    match graphviz_proc.wait_with_output() {
        Ok(output) => {
            // process can exit cleanly on error
            if output.status.success() {
                base64::engine::general_purpose::STANDARD.encode(&output.stdout[..])
            } else {
                panic!("{}", String::from_utf8_lossy(&output.stderr[..]));
            }
        }
        Err(e) => panic!("{}", e),
    }
}

pub fn evcxr_render_as(mime: &str, content: String) {
    println!(
        "EVCXR_BEGIN_CONTENT {}\n{}\nEVCXR_END_CONTENT",
        mime, content
    );
}
