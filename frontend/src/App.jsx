// ============================================================================
// Imports
// ============================================================================

// React utilities used to create shared context and manage component state.
import { createContext, useState } from "react";

// Global styles for main application
import "./App.css";

// Shared page layouts and workflow components
import Header from "./Header";
import ProgressSteps from "./ProgressSteps";
import SettingsSection from "./SettingsSection";
import UploadCard from "./UploadCard";
import ResultsPanel from "./ResultsPanel";
import VerifTypePanel from "./VerifTypePanel.jsx";
import FilePreview from "./FilePreview";
import NetworkExplorer from "./NetworkExplorer";

// Simulink-specific workflow
import Simulink from "./simulink/Simulink.jsx";

/**
 * Provides current workflow step and selected verification type to components
 * for the Simulink workflow
 */
export const slContext = createContext(null);

// ============================================================================
// Backend configuration
// ============================================================================

/**
 * Root URL for the local FastAPI backend.
 * 
 * Individual endpoint paths are added later by getEndpoint() based on user
 * selection.
 */
const API_BASE_URL = "http://localhost:8000";

// ============================================================================
// Main application component
// ============================================================================


function App() {


    // ============================================================================
    // State tracking for workflow
    // ============================================================================

    // Stores response returned by the verification backend such as successful
    // and error messages.
    const [backendStatus, setBackendStatus] = useState(null);
    const [requestError, setRequestError] = useState("");

    // Stores the user-defined configuration settings that will be used in the
    // verification worksflow.
    const [settings, setSettings] = useState({
        nnFormat: "ONNX",
        probType: "SAFETY",
        absRequired: "NO",
        numOfAbsNodes: 75,
        technique: "MILP",
        lastRelu: "NO",
        K: 1,
        absType: "INTERVAL",
        partitionType: "FIXED",
        solverType: "Gurobi"
    });

    // Tracking values used for determining which React state will be displayed
    // at any given time.
    const [verificationType, setVerificationType] = useState(null);
    const [currentStep, setCurrentStep] = useState(1);

    // Upload file states based on user uploads
    const [networkFile, setNetworkFile] = useState(null);
    const [propertyFile, setPropertyFile] = useState(null);
    const [dynamicsFile, setDynamicsFile] = useState(null);
    const [propertyType, setPropertyType] = useState("VNNLIB");

    // State tracking for whether a verification is currently running.
    // Used for button lockout to prevent spamming runs.
    const [isVerifying, setIsVerifying] = useState(false);

    // Flags that help track verification type for workflow purposes
    const isAverinnType =
        verificationType === "nn" ||
        verificationType === "nncs" ||
        verificationType === "hybrid";

    const isSimulinkType =
        verificationType === "simulink-system";

    // Used in conjuction with upload cards to determine if dynamics are expected
    // to be provided by user.
    const requiresDynamicsFile =
        verificationType === "nncs" ||
        verificationType === "hybrid";

    // Navigation information made available through slContext
    const contextData = {
        currentStep,
        setCurrentStep,
        currentType: verificationType,
        setCurrentType: setVerificationType,
        API_BASE_URL
    };

    // State trackers for file previews
    const [previewOpen, setPreviewOpen] = useState(false);
    const [previewFile, setPreviewFile] = useState(null);
    const [previewTitle, setPreviewTitle] = useState("");

    // State used by the uploaded ONNX Network Explorer.
    const [networkExplorerOpen, setNetworkExplorerOpen] = useState(false);
    const [networkSummary, setNetworkSummary] = useState(null);
    const [networkExplorerError, setNetworkExplorerError] = useState("");
    const [isLoadingNetwork, setIsLoadingNetwork] = useState(false);
    const [selectedNetworkLayerId, setSelectedNetworkLayerId] = useState(0);

    // ============================================================================
    // Workflow reset and nav helpers
    // ============================================================================

    // Clears data associated with previous verification attempt
    function resetVerificationState() {
        setBackendStatus(null);
        setRequestError("");
        setNetworkFile(null);
        setPropertyFile(null);
        setDynamicsFile(null);
        setNetworkExplorerOpen(false);
        setNetworkSummary(null);
        setNetworkExplorerError("");
        setIsLoadingNetwork(false);
        setSelectedNetworkLayerId(0);
    }

    // Handles verification type selection from the VerifTypePanel
    // Clears data from previous attemped verifications and advances
    // supported workflows to Step 2
    function handleVerificationTypeSelection(type) {
        setVerificationType(type);
        resetVerificationState();

        if (
            type === "nn" ||
            type === "nncs" ||
            type === "hybrid" ||
            type === "simulink-system"
        ) {
            setCurrentStep(2);
        }
    }

    // Function to check uploaded file types to make sure they posses the correct
    // format extension.
    function validateCurrentFiles() {
        if (!networkFile) {
            return "Please upload a neural-network file.";
        }

        if (!propertyFile) {
            return "Please upload a VNNLIB property file.";
        }

        if (requiresDynamicsFile && !dynamicsFile) {
            return verificationType === "hybrid"
                ? "Please upload a YAML hybrid-dynamics file."
                : "Please upload an INI dynamics file.";
        }

        return "";
    }

    // Helper function for uploaded file preview
    function openPreview(file, title) {
        setPreviewFile(file);
        setPreviewTitle(title);
        setPreviewOpen(true);
    }


    /**
     * Upload the selected ONNX file to the visualization endpoint and open
     * the explorer with the normalized architecture returned by FastAPI.
     */
    async function openNetworkExplorer() {
        if (!networkFile) return;

        setNetworkExplorerOpen(true);
        setNetworkSummary(null);
        setNetworkExplorerError("");
        setIsLoadingNetwork(true);
        setSelectedNetworkLayerId(0);

        try {
            if (!networkFile.name.toLowerCase().endsWith(".onnx")) {
                throw new Error(
                    "The Network Explorer currently supports ONNX files only."
                );
            }

            const formData = new FormData();
            formData.append("network_file", networkFile);

            const response = await fetch(
                `${API_BASE_URL}/visualize-network`,
                {
                    method: "POST",
                    body: formData
                }
            );

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    typeof data.detail === "string"
                        ? data.detail
                        : "The backend could not inspect this network."
                );
            }

            setNetworkSummary(data);
            setSelectedNetworkLayerId(data.layers?.[0]?.id ?? 0);
        } catch (error) {
            setNetworkExplorerError(
                error instanceof Error
                    ? error.message
                    : "Unable to open the Network Explorer."
            );
        } finally {
            setIsLoadingNetwork(false);
        }
    }

    // Takes user to Step 3 (Settings page) after Step 2 (Upload page)
    function goToSettings() {
        const validationError = validateCurrentFiles();

        if (validationError) {
            setRequestError(validationError);
            return;
        }

        setRequestError("");
        setCurrentStep(3);
    }

    // Retrieves endpoint based on verification type chosen by user
    function getEndpoint() {
        if (verificationType === "nn") {
            return `${API_BASE_URL}/run-nn-averinn`;
        }

        if (verificationType === "nncs") {
            return `${API_BASE_URL}/run-nncs-averinn`;
        }

        if (verificationType === "hybrid") {
            return `${API_BASE_URL}/run-hybrid-averinn`;
        }

        throw new Error(
            `Unsupported verification type: ${verificationType}`
        );
    }

    // Hybrid receives a reduced payload because its backend currently uses 
    // only lastRelu and K.
    function getSubmittedSettings() {
        if (verificationType === "hybrid") {
            return {
                lastRelu: settings.lastRelu,
                K: Number(settings.K)
            };
        }

        return {
            ...settings,
            specformat: propertyType
        };
    }

    // ============================================================================
    // Verification Request
    // ============================================================================

    /**
     * Submits the selected files and setting to the appropriate FastAPI
     * verification endpoint.
     * 
     * Workflow:
     * - Prevent duplicate submissions
     * - Validate required files
     * - Lock interface (prevents button-spamming)
     * - Create a multipart FormData request
     * - Add serialized setting and uploaded files
     * - POST the request to the selected backend endpoint
     * - Parse and validate the backend response
     * - Store successful results and advance to Step 4 (Results page)
     * - Display errors if applicable
     * - Re-enable the interface regardless of outcome
     */
    async function callAPI() {

        // Button is already disabled when clicked but this is added layer of protection
        // against the function receiving duplicate calls.
        if (isVerifying) {
            return;
        }

        const validationError = validateCurrentFiles();

        if (validationError) {
            setRequestError(validationError);
            return;
        }

        setIsVerifying(true);
        setRequestError("");
        setBackendStatus(null);

        try {

            // Construction of FormData for converting into JSON string.
            const formData = new FormData();

            formData.append(
                "settings",
                JSON.stringify(getSubmittedSettings())
            );

            formData.append("network_file", networkFile);
            formData.append("property_file", propertyFile);

            if (requiresDynamicsFile) {
                formData.append("dynamics_file", dynamicsFile);
            }

            const response = await fetch(getEndpoint(), {
                method: "POST",
                body: formData
            });

            // Parse the backend response as JSON.
            const data = await response.json();

            if (!response.ok) {
                const detail =
                    typeof data.detail === "string"
                        ? data.detail
                        : "The backend rejected the verification request.";

                throw new Error(detail);
            }

            setBackendStatus(data);
            setCurrentStep(4);

        } catch (error) {
            console.error(error);

            setRequestError(
                error instanceof Error
                    ? error.message
                    : "Unable to complete verification."
            );
        } finally {
            setIsVerifying(false);
        }
    }

    // Returns page title depending on workflow type
    function getPageTitle() {
        if (verificationType === "hybrid") {
            return "Neural Network Controlled Hybrid Dynamical Systems";
        }

        if (verificationType === "nncs") {
            return "Neural Network Controlled Linear Systems";
        }

        return "Neural Network Property Checking";
    }

    // Builds upload page requirements depending on verification type
    function getDynamicsCardConfiguration() {
        if (verificationType === "hybrid") {
            return {
                title: "Hybrid Dynamics File",
                formatValue: "YAML",
                formatOptions: ["YAML"],
                acceptedFileTypes: ".yaml,.yml",
                primaryButtonText: "View Dynamics",
                secondaryButtonText: "Validate Dynamics"
            };
        }

        return {
            title: "Dynamics File",
            formatValue: "INI",
            formatOptions: ["INI"],
            acceptedFileTypes: ".ini",
            primaryButtonText: "View Dynamics",
            secondaryButtonText: "Validate Dynamics"
        };
    }

    // Computes dynamics-card configuration so that Step 2 (Upload page)
    // displays dynamics uploader if nncs or hybrid
    const dynamicsCard = getDynamicsCardConfiguration();

    // ============================================================================
    // Main Application Render
    // ============================================================================

    return (

        // Makes the current workflow step and verification type
        // available to descendant components that use slContext.
        <slContext.Provider value={contextData}>
            <div className="background d-flex flex-column" style={{minHeight:"100svh"}}>

                {/* ============================================================
                    Shared Page Header
                ============================================================ */}
                <Header />

                {/* ============================================================
                    Shared Progression Display
                ============================================================ */}
                <ProgressSteps
                    currentStep={currentStep}
                    setCurrentStep={setCurrentStep}
                />

                {/* ============================================================
                    Verification Type Selection Page
                ============================================================ */}
                {currentStep === 1 && (
                    <div>
                        <h1 className="h3 fw-bold text-center">
                            Select Verification Type
                        </h1>

                        <VerifTypePanel
                            onSelectType={
                                handleVerificationTypeSelection
                            }
                        />
                    </div>
                )}

                {/* ============================================================
                    Simulink Verification Workflow
                ============================================================ */}
                {isSimulinkType && currentStep > 1 && (
                    <Simulink />
                )}

                {/* ============================================================
                    Upload Page
                ============================================================ */}
                {isAverinnType && currentStep === 2 && (
                    <section className="container py-4">
                        <h1 className="h3 fw-bold">
                            {getPageTitle()}
                        </h1>

                        <p className="fs-5">
                            Provide the required model, property,
                            and dynamics files to continue.
                        </p>

                        {requestError && (
                            <div
                                className="alert alert-danger"
                                role="alert"
                            >
                                {requestError}
                            </div>
                        )}

                        {/* ============================================================
                            File Upload Cards
                        ============================================================ */}
                        <div className="row g-4 mt-3">
                            
                            {/* Neural Network Upload Card */}
                            <div
                                // Regular NN only uses two cards so displays smaller format
                                className={
                                    requiresDynamicsFile
                                        ? "col-12 col-lg-4"
                                        : "col-12 col-lg-6"
                                }
                            >
                                <UploadCard
                                    title="Neural Network"
                                    formatLabel="Format"
                                    formatValue={
                                        verificationType === "hybrid"
                                            ? "ONNX"
                                            : settings.nnFormat
                                    }
                                    formatOptions={
                                        verificationType === "hybrid"
                                            ? ["ONNX"]
                                            : [
                                                "ONNX",
                                                "SHERLOCK",
                                                "ISHERLOCK",
                                                "NNET"
                                            ]
                                    }
                                    onFormatChange={(newFormat) =>
                                        setSettings(
                                            (previousSettings) => ({
                                                ...previousSettings,
                                                nnFormat: newFormat
                                            })
                                        )
                                    }
                                    fileLabel="Model file"
                                    file={networkFile}
                                    onFileChange={setNetworkFile}
                                    acceptedFileTypes={
                                        verificationType === "hybrid"
                                            ? ".onnx"
                                            : ".onnx,.nnet,.sherlock,.isherlock"
                                    }
                                    primaryButtonText="View Network"
                                    onPrimaryButtonClick={openNetworkExplorer}
                                    secondaryButtonText="Validate Network"
                                />
                            </div>

                            {/* Specification File Upload Card */}
                            <div
                                className={
                                    requiresDynamicsFile
                                        ? "col-12 col-lg-4"
                                        : "col-12 col-lg-6"
                                }
                            >
                                <UploadCard
                                    title="Property Specification"
                                    formatLabel="Specification type"
                                    formatValue={propertyType}
                                    formatOptions={["VNNLIB"]}
                                    onFormatChange={setPropertyType}
                                    fileLabel="Property file"
                                    file={propertyFile}
                                    onFileChange={setPropertyFile}
                                    acceptedFileTypes=".vnnlib"
                                    primaryButtonText="View Property"
                                    onPrimaryButtonClick={() =>
                                        openPreview(
                                            propertyFile,
                                            "Property Specification"
                                        )
                                    }
                                    secondaryButtonText="Validate Property"
                                />
                            </div>

                            {/* Dynamics Sheet Upload Card */}
                            {requiresDynamicsFile && (
                                <div className="col-12 col-lg-4">
                                    <UploadCard
                                        title={dynamicsCard.title}
                                        formatLabel="Format"
                                        formatValue={dynamicsCard.formatValue}
                                        formatOptions={dynamicsCard.formatOptions}
                                        onFormatChange={() => {}}
                                        fileLabel="Dynamics file"
                                        file={dynamicsFile}
                                        onFileChange={setDynamicsFile}
                                        acceptedFileTypes={
                                            dynamicsCard.acceptedFileTypes
                                        }
                                        primaryButtonText={
                                            dynamicsCard.primaryButtonText
                                        }
                                        onPrimaryButtonClick={() =>
                                            openPreview(
                                                dynamicsFile,
                                                dynamicsCard.title
                                            )
                                        }
                                        secondaryButtonText={
                                            dynamicsCard.secondaryButtonText
                                        }
                                    />
                                </div>
                            )}
                        </div>

                        
                        {/* ============================================================
                            Navigation Panel for Upload Page (Previous/Next)
                        ============================================================ */}
                        <hr className="section-divider" />

                        <div className="bottom-panel d-flex justify-content-between">
                            <button
                                onClick={() => setCurrentStep(1)}
                                className="btn btn-primary px-4"
                            >
                                Previous Page
                            </button>

                            <button
                                onClick={goToSettings}
                                className="btn btn-primary px-4"
                            >
                                Continue to Settings
                            </button>
                        </div>
                    </section>
                )}

                {/* ============================================================
                    Settings Configuration Page
                ============================================================ */}
                {isAverinnType && currentStep === 3 && (
                    <section className="container py-4">
                        
                        {/* Page display layout dependin on Hybrid vs other verifications */}
                        {verificationType === "hybrid" ? (
                            <HybridSettings
                                settings={settings}
                                setSettings={setSettings}
                                networkFile={networkFile}
                                propertyFile={propertyFile}
                                dynamicsFile={dynamicsFile}
                            />
                        ) : (
                            <SettingsSection
                                settings={settings}
                                setSettings={setSettings}
                                verificationType={verificationType}
                                networkFile={networkFile}
                                propertyType={propertyType}
                                propertyFile={propertyFile}
                                dynamicsFile={dynamicsFile}
                            />
                        )}

                        {requestError && (
                            <div
                                className="alert alert-danger"
                                role="alert"
                            >
                                {requestError}
                            </div>
                        )}

                        <hr className="section-divider" />

                        {/* ============================================================
                            Navigation Panel for Settings Page (Previous/Start Verification)
                        ============================================================ */}
                        <div className="bottom-panel d-flex justify-content-between">
                            <button
                                onClick={() => setCurrentStep(2)}
                                className="btn btn-primary px-4"
                            >
                                Previous Page
                            </button>

                            <button
                                onClick={callAPI}
                                disabled={isVerifying}
                                className="btn btn-primary px-4"
                            >
                                {isVerifying
                                    ? "Running Verification..."
                                    : "Start Verification"}
                            </button>
                        </div>
                    </section>
                )}

                {/* ============================================================
                    Results Panel
                ============================================================ */}
                {isAverinnType && currentStep === 4 && (
                    <section className="container py-4">
                        <h1 className="h3 fw-bold text-center">
                            Analysis Results
                        </h1>

                        {/* Displays payload from backend verification */}
                        <ResultsPanel
                            backendStatus={backendStatus}
                        />

                        <hr className="section-divider" />

                        {/* Button to return to previous page */}
                        <div className="bottom-panel d-flex justify-content-start">
                            <button
                                onClick={() => setCurrentStep(3)}
                                className="btn btn-primary px-4"
                            >
                                Previous Page
                            </button>
                        </div>
                    </section>
                )}
            </div>

            {/* ============================================================
                    Uploaded File Preview
                ============================================================ */}
                <FilePreview
                    file={previewFile}
                    title={previewTitle}
                    isOpen={previewOpen}
                    onClose={() => setPreviewOpen(false)}
                />

                <NetworkExplorer
                    isOpen={networkExplorerOpen}
                    network={networkSummary}
                    selectedLayerId={selectedNetworkLayerId}
                    onSelectLayer={setSelectedNetworkLayerId}
                    isLoading={isLoadingNetwork}
                    error={networkExplorerError}
                    onClose={() => setNetworkExplorerOpen(false)}
                    networkFile={networkFile}
                    apiBaseUrl={API_BASE_URL}
                />
        </slContext.Provider>
    );
}


// ============================================================================
// Hybrid Verification Settings Component
// ============================================================================

/**
 * Displays simplified setting page when using Hybrid workflow.
 */
function HybridSettings({
    settings,
    setSettings,
    networkFile,
    propertyFile,
    dynamicsFile
}) {
    return (
        <section className="container py-4">
            <h1 className="h3 fw-bold">
                Hybrid Verification Settings
            </h1>

            <p className="fs-5">
                Configure the Hybrid AVERINN reachability run.
            </p>

            <div className="row g-4 mt-3">
                <div className="col-12 col-lg-6">
                    <div className="verification-card">
                        <h2 className="verification-card-title">
                            Reachability Settings
                        </h2>

                        <div className="form-row-custom">
                            <label>Last layer ReLU</label>

                            <select
                                className="form-select"
                                value={settings.lastRelu}
                                onChange={(event) =>
                                    setSettings(
                                        (previousSettings) => ({
                                            ...previousSettings,
                                            lastRelu:
                                                event.target.value
                                        })
                                    )
                                }
                            >
                                <option value="NO">No</option>
                                <option value="YES">Yes</option>
                            </select>
                        </div>

                        <div className="form-row-custom">
                            <label>Number of iterations</label>

                            <input
                                className="form-control"
                                type="number"
                                min="1"
                                step="1"
                                value={settings.K}
                                onChange={(event) =>
                                    setSettings(
                                        (previousSettings) => ({
                                            ...previousSettings,
                                            K: Number(
                                                event.target.value
                                            )
                                        })
                                    )
                                }
                            />
                        </div>
                    </div>
                </div>

                <div className="col-12 col-lg-6">
                    <div className="verification-card">
                        <h2 className="verification-card-title">
                            Selected Inputs
                        </h2>

                        <SelectedHybridInput
                            label="Verification type"
                            value="NN-controlled hybrid system"
                        />

                        <SelectedHybridInput
                            label="Network"
                            value={networkFile?.name ?? "Not selected"}
                        />

                        <SelectedHybridInput
                            label="Property"
                            value={propertyFile?.name ?? "Not selected"}
                        />

                        <SelectedHybridInput
                            label="Dynamics"
                            value={dynamicsFile?.name ?? "Not selected"}
                        />
                    </div>
                </div>
            </div>
        </section>
    );
}


// ============================================================================
// Selected Hybrid Input Row
// ============================================================================
function SelectedHybridInput({ label, value }) {
    return (
        <div className="selected-input-row">
            <span>{label}</span>
            <span>...........</span>
            <span>{value}</span>
        </div>
    );
}


export default App;
