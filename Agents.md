## Security Rules

- Only read, write, search, and process files inside this project folder.
- Do not inspect the user's home directory, Desktop, Downloads, Documents, or any folder outside this repository.
- Do not recursively scan parent directories.
- Process source documents only from `data/input/`.
- Do not read or print `.env` values.
- Never expose API keys, tokens, credentials, or private document contents.
- Use `.env.example` only to understand required environment variable names.
- Never commit secrets, raw API keys, credentials, tokens, parsed private files, or local database outputs.