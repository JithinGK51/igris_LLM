use super::Tokenizer;
use std::collections::HashMap;

/// Task-aware tokenizer that wraps any tokenizer and adds task prefix tokens
pub struct TaskAwareTokenizer<T: Tokenizer> {
    inner: T,
    task_tokens: HashMap<String, usize>,
    task_token_strings: HashMap<String, &'static str>,
}

impl<T: Tokenizer> TaskAwareTokenizer<T> {
    /// Create a new task-aware tokenizer wrapping the given tokenizer
    pub fn new(inner: T) -> Self {
        let mut task_tokens = HashMap::new();
        let mut task_token_strings = HashMap::new();

        // Reserve first 3 token IDs for task tokens
        task_tokens.insert("text".to_string(), 0);
        task_tokens.insert("code".to_string(), 1);
        task_tokens.insert("image".to_string(), 2);

        task_token_strings.insert("text".to_string(), "<TASK_TEXT>");
        task_token_strings.insert("code".to_string(), "<TASK_CODE>");
        task_token_strings.insert("image".to_string(), "<TASK_IMAGE>");

        Self {
            inner,
            task_tokens,
            task_token_strings,
        }
    }

    /// Get the task token ID for a given task
    pub fn get_task_token_id(&self, task: &str) -> Option<usize> {
        self.task_tokens.get(task).copied()
    }

    /// Get the task token string for a given task
    pub fn get_task_token_string(&self, task: &str) -> Option<&'static str> {
        self.task_token_strings.get(task).copied()
    }

    /// Tokenize text with a task prefix
    pub fn tokenize_with_task(&self, task: &str, text: &str) -> Vec<usize> {
        let task_token = self.get_task_token_string(task).unwrap_or("<TASK_TEXT>");
        let prefixed_text = format!("{} {}", task_token, text);
        self.inner.tokenize(&prefixed_text)
    }

    /// Untokenize tokens, removing task prefix if present
    pub fn untokenize_removing_task(&self, tokens: &[usize]) -> String {
        let text = self.inner.untokenize(tokens);
        
        // Remove task prefix if present
        if text.starts_with("<TASK_TEXT> ") {
            text.strip_prefix("<TASK_TEXT> ").unwrap_or(&text).to_string()
        } else if text.starts_with("<TASK_CODE> ") {
            text.strip_prefix("<TASK_CODE> ").unwrap_or(&text).to_string()
        } else if text.starts_with("<TASK_IMAGE> ") {
            text.strip_prefix("<TASK_IMAGE> ").unwrap_or(&text).to_string()
        } else {
            text
        }
    }

    /// Get the inner tokenizer (for direct access if needed)
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get mutable access to the inner tokenizer
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Check if a token is a task token
    pub fn is_task_token(&self, token_id: usize) -> bool {
        token_id < 3 // First 3 tokens are reserved for tasks
    }

    /// Get the task type for a given task token ID
    pub fn get_task_for_token_id(&self, token_id: usize) -> Option<&str> {
        match token_id {
            0 => Some("text"),
            1 => Some("code"),
            2 => Some("image"),
            _ => None,
        }
    }
}

impl<T: Tokenizer> Tokenizer for TaskAwareTokenizer<T> {
    fn vocab_size(&self) -> usize {
        // Add 3 for task tokens
        self.inner.vocab_size() + 3
    }

    fn tokenize(&self, text: &str) -> Vec<usize> {
        // If text already contains task prefix, use as-is
        if text.starts_with("<TASK_") {
            self.inner.tokenize(text)
        } else {
            // Default to text task if no task specified
            self.tokenize_with_task("text", text)
        }
    }

    fn untokenize(&self, tokens: &[usize]) -> String {
        self.inner.untokenize(tokens)
    }
}

/// Extension trait for easy conversion to task-aware tokenizer
pub trait ToTaskAware {
    fn to_task_aware(self) -> TaskAwareTokenizer<Self>
    where
        Self: Tokenizer + Sized,
    {
        TaskAwareTokenizer::new(self)
    }
}

impl<T: Tokenizer> ToTaskAware for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::SimpleTokenizer;

    #[test]
    fn test_task_aware_tokenizer_creation() {
        let simple_tokenizer = SimpleTokenizer::new("Hello world");
        let task_tokenizer = TaskAwareTokenizer::new(simple_tokenizer);

        assert_eq!(task_tokenizer.get_task_token_id("text"), Some(0));
        assert_eq!(task_tokenizer.get_task_token_id("code"), Some(1));
        assert_eq!(task_tokenizer.get_task_token_id("image"), Some(2));
        assert_eq!(task_tokenizer.get_task_token_id("invalid"), None);
    }

    #[test]
    fn test_task_token_strings() {
        let simple_tokenizer = SimpleTokenizer::new("Hello world");
        let task_tokenizer = TaskAwareTokenizer::new(simple_tokenizer);

        assert_eq!(task_tokenizer.get_task_token_string("text"), Some("<TASK_TEXT>"));
        assert_eq!(task_tokenizer.get_task_token_string("code"), Some("<TASK_CODE>"));
        assert_eq!(task_tokenizer.get_task_token_string("image"), Some("<TASK_IMAGE>"));
    }

    #[test]
    fn test_tokenize_with_task() {
        let simple_tokenizer = SimpleTokenizer::new("Hello world");
        let task_tokenizer = TaskAwareTokenizer::new(simple_tokenizer);

        let tokens = task_tokenizer.tokenize_with_task("text", "Hello world");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_task_token_detection() {
        let simple_tokenizer = SimpleTokenizer::new("Hello world");
        let task_tokenizer = TaskAwareTokenizer::new(simple_tokenizer);

        assert!(task_tokenizer.is_task_token(0));
        assert!(task_tokenizer.is_task_token(1));
        assert!(task_tokenizer.is_task_token(2));
        assert!(!task_tokenizer.is_task_token(3));
    }

    #[test]
    fn test_get_task_for_token_id() {
        let simple_tokenizer = SimpleTokenizer::new("Hello world");
        let task_tokenizer = TaskAwareTokenizer::new(simple_tokenizer);

        assert_eq!(task_tokenizer.get_task_for_token_id(0), Some("text"));
        assert_eq!(task_tokenizer.get_task_for_token_id(1), Some("code"));
        assert_eq!(task_tokenizer.get_task_for_token_id(2), Some("image"));
        assert_eq!(task_tokenizer.get_task_for_token_id(3), None);
    }

    #[test]
    fn test_extension_trait() {
        let simple_tokenizer = SimpleTokenizer::new("Hello world");
        let task_tokenizer = simple_tokenizer.to_task_aware();
        
        assert_eq!(task_tokenizer.get_task_token_id("text"), Some(0));
    }
}
