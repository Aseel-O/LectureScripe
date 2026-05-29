/**
 * Global TypeScript Type Declarations
 *
 * Declares global module types for CSS imports and other global modules.
 */

// Enable TypeScript support for CSS module imports
// Allows: import './styles.css' without type errors
declare module '*.css';