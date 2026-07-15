
import { useEffect, useState } from "react";
import "./FilePreview.css";

// ============================================================================
// File Preview Modal Component
// ============================================================================

/**
 * Displays the contents of an uploaded text-based input file in a
 * read-only modal window.
 *
 * This component is shared by multiple upload cards (currently the
 * Property Specification and Dynamics uploads) so the same preview
 * interface can be reused regardless of which file type is selected.
 *
 * Workflow:
 * - Wait until the preview modal is opened.
 * - Read the uploaded File object using the browser File API.
 * - Store the file contents in component state.
 * - Display the contents inside a scrollable <pre> block so original
 *   formatting and whitespace are preserved.
 * - Allow the user to dismiss the preview by clicking the background,
 *   the close button, or the "Close" action button.
 *
 * Note:
 * This component is intended for plain-text files such as VNNLIB,
 * INI, YAML, and similar formats. Binary files (such as ONNX neural
 * networks) require a specialized viewer and are handled separately.
 */

function FilePreview({ file, title, isOpen, onClose }) {

    const [contents, setContents] = useState("");

    useEffect(() => {
        if (!file || !isOpen) return;

        async function loadFile() {
            try {
                const text = await file.text();
                setContents(text);
            }
            catch {
                setContents("Unable to preview this file.");
            }
        }

        loadFile();
    }, [file, isOpen]);

    if (!isOpen) return null;

    return (
        <div className="preview-backdrop" onClick={onClose}>
            <div
                className="preview-modal"
                onClick={(e) => e.stopPropagation()}
            >
                <div className="preview-header">
                    <h3>{title}</h3>

                    <button onClick={onClose}>✕</button>
                </div>

                <pre className="preview-content">
                    {contents}
                </pre>

                <div className="preview-footer">
                    <button
                        className="btn btn-primary"
                        onClick={onClose}
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
}

export default FilePreview;