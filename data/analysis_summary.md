# Part C measured results

## Test-set comparison

| Tokenizer | Tokens | Tokens/word | Bytes/token |
|---|---:|---:|---:|
| ours | 882,054 | 1.186 | 7.040 |
| gpt2 | 4,443,678 | 5.976 | 1.397 |
| o200k_base | 1,214,076 | 1.633 | 5.114 |

## Merge findings

- Invalid standalone UTF-8 merge tokens: 106
- First space-prefixed whole Urdu word: merge 32 / token 288 / ` کی`
- The JSON file contains merges 1–50, each 5,000th merge, early letter reconstruction, and the 20 longest tokens.
