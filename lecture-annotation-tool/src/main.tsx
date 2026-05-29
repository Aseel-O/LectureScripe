/**
 * React Application Entry Point
 *
 * Initializes the React application and mounts it to the DOM.
 * Uses React 18's StrictMode for development checks and warnings.
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// Mount the root React component to the #root DOM element
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
