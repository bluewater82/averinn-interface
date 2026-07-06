import "./UploadCard.css";

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
    secondaryButtonText
}) {
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
                        <option key={option}>{option}</option>
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
                <button className="btn btn-outline-primary">
                    {primaryButtonText}
                </button>

                <button className="btn btn-outline-primary">
                    {secondaryButtonText}
                </button>
            </div>
        </div>
    );
}

export default UploadCard;