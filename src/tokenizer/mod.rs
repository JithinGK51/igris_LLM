mod byte;
pub use byte::*;

mod simple;
pub use simple::*;

mod sentencepiece;
pub use sentencepiece::*;

mod task_aware;
pub use task_aware::*;

pub trait Tokenizer {
    fn vocab_size(&self) -> usize;
    fn tokenize(&self, string: &str) -> Vec<usize>;
    fn untokenize(&self, tokens: &[usize]) -> String;
}
