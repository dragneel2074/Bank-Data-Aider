/**
 * Utility functions for string manipulation
 */

export function toTitleCase(str) {
    return str.split(' ')
             .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
             .join(' ');
} 