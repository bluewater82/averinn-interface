import "./UploadCard.css";

/**
 * Reusable upload card used throughout the verification setup workflow.
 *
 * Rather than creating separate components for each upload step,
 * this component is configured entirely through props (title,
 * labels, dropdown options, accepted file types, etc.). This
 * allows the same UI structure to support multiple upload stages
 * while keeping all upload behavior consistent.
 */

function UploadCard({
    title,
    formatLabel,
    formatValue,
    formatOptions,
    onFormatChange,
    fileLabel,
    file,
    onFileChange,
    acceptedFileTypes,
    primaryButtonText,
    secondaryButtonText,
    onPrimaryButtonClick
}) {

    /**
     * Called whenever the user selects a file from the browser.
     *
     * The browser provides all selected files in e.target.files.
     * Since this interface only accepts a single file, we retrieve
     * the first entry and pass it back to the parent component,
     * which stores the file in application state.
     */
    function handleFileInput(e) {
        const selectedFile = e.target.files[0];

        if (selectedFile) {
            onFileChange(selectedFile);
        }
    }

    return (
        <div className="upload-card">
            <h2 className="upload-card-title">{title}</h2>

            <div className="upload-field">
                <label>{formatLabel}</label>

                <select
                    className="form-select"
                    value={formatValue}
                    onChange={(e) => onFormatChange(e.target.value)}
                >
                    {formatOptions.map((option) => (
                        <option key={option} value={option}>
                            {option}
                        </option>
                    ))}
                </select>
            </div>

            <div className="upload-field">
                <label>{fileLabel}</label>

                <label className="upload-dropzone">
                    <input
                        type="file"
                        accept={acceptedFileTypes}
                        onChange={handleFileInput}
                        hidden
                    />

                    <span className="upload-icon">↑</span>

                    <span>
                        {file
                            ? file.name
                            : "Click to upload or drag and drop"}
                    </span>
                </label>
            </div>

            <div className="upload-actions">
                <button className="btn btn-outline-primary"
                    onClick={onPrimaryButtonClick}
                    disabled={!file}
                >
                    {primaryButtonText}
                </button>

                
            </div>
        </div>
    );
}

export default UploadCard;