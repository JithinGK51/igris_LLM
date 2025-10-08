use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Error as IoError};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalDataset {
    pub task: String,
    pub instruction: String,
    pub input: String,
    pub output: String,
}

impl MultiModalDataset {
    /// Create a new MultiModalDataset entry
    pub fn new(task: String, instruction: String, input: String, output: String) -> Self {
        Self {
            task,
            instruction,
            input,
            output,
        }
    }

    /// Combine instruction, input, and output into a single training sequence
    pub fn to_training_sequence(&self) -> String {
        let mut sequence = String::new();
        
        // Add instruction
        if !self.instruction.is_empty() {
            sequence.push_str(&self.instruction);
            sequence.push('\n');
        }
        
        // Add input if present
        if !self.input.is_empty() {
            sequence.push_str(&self.input);
            sequence.push('\n');
        }
        
        // Add output
        sequence.push_str(&self.output);
        
        sequence
    }

    /// Get the task token ID for this dataset entry
    pub fn task_token_id(&self) -> usize {
        match self.task.as_str() {
            "text" => 0,
            "code" => 1,
            "image" => 2,
            _ => 0, // Default to text task
        }
    }

    /// Get the task token string for this dataset entry
    pub fn task_token(&self) -> &'static str {
        match self.task.as_str() {
            "text" => "<TASK_TEXT>",
            "code" => "<TASK_CODE>",
            "image" => "<TASK_IMAGE>",
            _ => "<TASK_TEXT>", // Default to text task
        }
    }
}

/// Load a JSONL dataset file and parse it into MultiModalDataset entries
pub fn load_jsonl<P: AsRef<Path>>(path: P) -> Result<Vec<MultiModalDataset>, IoError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut datasets = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue; // Skip empty lines
        }

        match serde_json::from_str::<MultiModalDataset>(&line) {
            Ok(dataset) => datasets.push(dataset),
            Err(e) => {
                eprintln!("Warning: Failed to parse line {}: {}", line_num + 1, e);
                eprintln!("Line content: {}", line);
                continue;
            }
        }
    }

    Ok(datasets)
}

/// Load a plain text dataset (backward compatibility)
pub fn load_plain_text<P: AsRef<Path>>(path: P) -> Result<String, IoError> {
    std::fs::read_to_string(path)
}

/// Convert a plain text dataset to MultiModalDataset format
pub fn text_to_multimodal(text: &str, task: &str) -> Vec<MultiModalDataset> {
    // Split text into chunks (simple approach - can be improved)
    let chunks: Vec<&str> = text.split('\n').filter(|s| !s.trim().is_empty()).collect();
    
    chunks.into_iter().map(|chunk| {
        MultiModalDataset::new(
            task.to_string(),
            "Continue the text:".to_string(),
            "".to_string(),
            chunk.to_string(),
        )
    }).collect()
}

/// Validate that a dataset contains valid task types
pub fn validate_dataset(datasets: &[MultiModalDataset]) -> Result<(), String> {
    let valid_tasks = ["text", "code", "image"];
    
    for dataset in datasets {
        if !valid_tasks.contains(&dataset.task.as_str()) {
            return Err(format!("Invalid task type: '{}'. Valid types are: {:?}", 
                dataset.task, valid_tasks));
        }
    }
    
    Ok(())
}

/// Get statistics about a dataset
pub fn dataset_stats(datasets: &[MultiModalDataset]) -> DatasetStats {
    let mut task_counts = std::collections::HashMap::new();
    let mut total_chars = 0;
    let mut total_sequences = 0;

    for dataset in datasets {
        *task_counts.entry(dataset.task.clone()).or_insert(0) += 1;
        total_chars += dataset.to_training_sequence().chars().count();
        total_sequences += 1;
    }

    DatasetStats {
        total_sequences,
        total_characters: total_chars,
        task_distribution: task_counts,
        average_sequence_length: if total_sequences > 0 { 
            total_chars / total_sequences 
        } else { 
            0 
        },
    }
}

#[derive(Debug)]
pub struct DatasetStats {
    pub total_sequences: usize,
    pub total_characters: usize,
    pub task_distribution: std::collections::HashMap<String, usize>,
    pub average_sequence_length: usize,
}

impl std::fmt::Display for DatasetStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dataset Statistics:")?;
        writeln!(f, "  Total sequences: {}", self.total_sequences)?;
        writeln!(f, "  Total characters: {}", self.total_characters)?;
        writeln!(f, "  Average sequence length: {}", self.average_sequence_length)?;
        writeln!(f, "  Task distribution:")?;
        
        for (task, count) in &self.task_distribution {
            writeln!(f, "    {}: {} sequences", task, count)?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_multimodal_dataset_creation() {
        let dataset = MultiModalDataset::new(
            "text".to_string(),
            "Write a story".to_string(),
            "".to_string(),
            "Once upon a time...".to_string(),
        );

        assert_eq!(dataset.task, "text");
        assert_eq!(dataset.task_token_id(), 0);
        assert_eq!(dataset.task_token(), "<TASK_TEXT>");
    }

    #[test]
    fn test_training_sequence_generation() {
        let dataset = MultiModalDataset::new(
            "code".to_string(),
            "Write a Python function".to_string(),
            "def example():".to_string(),
            "    return 'hello'".to_string(),
        );

        let sequence = dataset.to_training_sequence();
        assert!(sequence.contains("Write a Python function"));
        assert!(sequence.contains("def example():"));
        assert!(sequence.contains("return 'hello'"));
    }

    #[test]
    fn test_jsonl_loading() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, r#"{{"task": "text", "instruction": "Write a story", "input": "", "output": "Once upon a time..."}}"#).unwrap();
        writeln!(temp_file, r#"{{"task": "code", "instruction": "Python function", "input": "", "output": "def hello(): pass"}}"#).unwrap();
        temp_file.flush().unwrap();

        let datasets = load_jsonl(temp_file.path()).unwrap();
        assert_eq!(datasets.len(), 2);
        assert_eq!(datasets[0].task, "text");
        assert_eq!(datasets[1].task, "code");
    }

    #[test]
    fn test_dataset_validation() {
        let valid_datasets = vec![
            MultiModalDataset::new("text".to_string(), "".to_string(), "".to_string(), "".to_string()),
            MultiModalDataset::new("code".to_string(), "".to_string(), "".to_string(), "".to_string()),
            MultiModalDataset::new("image".to_string(), "".to_string(), "".to_string(), "".to_string()),
        ];

        assert!(validate_dataset(&valid_datasets).is_ok());

        let invalid_datasets = vec![
            MultiModalDataset::new("invalid".to_string(), "".to_string(), "".to_string(), "".to_string()),
        ];

        assert!(validate_dataset(&invalid_datasets).is_err());
    }
}
