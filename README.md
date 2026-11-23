# Pennside

A full-stack project containing both frontend and backend modules.

## Project Structure

- `pennside_frontend/`: Frontend codebase (managed with pnpm)
- `pennside_backend/`: Backend codebase
- `README.md`: Project documentation

## Prerequisites

- Node.js
- pnpm
- Python

## Quick Start

### Frontend

```bash
# Navigate to the frontend directory
cd pennside_frontend

# Start the development server
pnpm dev
```

### Backend


```bash
# Navigate to the backend directory
cd pennside_backend

# Start the backend program
uvicorn interact_demo:app --reload --host 127.0.0.1 --port 8000
```