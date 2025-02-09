import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)



//npm create vite@latest ./ -- --template react
//npm run dev
//https://youtu.be/5fiXEGdEK10?si=a1jofqlyLNH_1qBR