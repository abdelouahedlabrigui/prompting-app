# Local LLM Interface

A React-based frontend for interacting with local Large Language Models (LLMs) through a clean, user-friendly interface.

## Features

- **Model Selection**: Choose from multiple LLM models (coding and conversation specialized)
- **Chat Interface**: Send prompts and receive responses from selected models
- **Conversation History**: View and manage previous interactions
- **Database Search**: Search and retrieve past conversations from backend database
- **Dark/Light Mode**: Toggle between themes for comfortable viewing
- **Responsive Design**: Works on desktop and mobile devices

## Models Available

| Model       | Category      | Specialization               |
|-------------|---------------|------------------------------|
| LlamaSharp  | Coding        | C# development               |
| CodeQwen    | Coding        | Python/JavaScript            |
| CodeLlama   | Coding        | Python/JavaScript            |
| Mistral     | Conversation  | General purpose              |
| Phi-3       | Conversation  | General purpose              |
| LLava       | Conversation  | Image-to-text processing     |

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```
3. Ensure you have the backend service running at `http://localhost:8000`

## Screenshot:
![alt text](https://raw.githubusercontent.com/abdelouahedlabrigui/prompting-app/refs/heads/main/images/llm-prompting-form.png)

![alt text](https://raw.githubusercontent.com/abdelouahedlabrigui/prompting-app/refs/heads/main/images/llms-interactions-search.png)


## Running the Application

```bash
npm run dev
# or
yarn dev
```

The application will be available at `http://localhost:3000`

## Project Structure

```
src/
├── app/
│   └── page.tsx        # Main application page
├── components/         # Reusable UI components
│   ├── llm-interface.tsx
│   ├── conversation-history.tsx
│   ├── model-selector.tsx
│   └── theme-toggle.tsx
└── lib/                # Utility functions and types
```

## Dependencies

- React
- Next.js
- shadcn/ui (for UI components)
- Lucide React (for icons)

## Backend Requirements

The frontend expects a backend service with these endpoints:

- `POST /{model-endpoint}` - For sending prompts to specific models
- `GET /interaction/{search-term}` - For searching conversation history

## Development Notes

- The application uses React hooks for state management
- TypeScript is used for type safety
- The UI is built with shadcn/ui components
- The interface is fully responsive

## Future Enhancements

- Add model configuration options
- Implement conversation saving/loading
- Add support for image uploads (for LLava model)
- Implement user authentication
