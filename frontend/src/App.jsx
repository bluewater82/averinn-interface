import { useState, createContext } from "react";
import "./App.css";
import Header from "./Header";
import ProgressSteps from "./ProgressSteps";
import SettingsSection from "./SettingsSection";
import UploadCard from "./UploadCard";
import ResultsPanel from "./ResultsPanel";
import Simulink from "./simulink/Simulink.jsx";
import VerifTypePanel from "./VerifTypePanel.jsx";

export const slContext = createContext(null);

function App() {
    const [backendStatus, setBackendStatus] = useState(null);

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

    const [dynamicsFile, setDynamicsFile] = useState(null);
    const [verificationType, setVerificationType] = useState(null);
    const [currentStep, setCurrentStep] = useState(1);

    const [networkFile, setNetworkFile] = useState(null);

    const [propertyType, setPropertyType] = useState("VNNLIB");
    const [propertyFile, setPropertyFile] = useState(null);

    const isAverinnType = verificationType === "nn" || verificationType === "nncs";
    const isSimulinkType = verificationType === "simulink-system";

    const contextData = {
        currentStep,
        setCurrentStep,
        currentType: verificationType,
        setCurrentType: setVerificationType
    };

    const [isVerifying, setIsVerifying] = useState(false);

    function handleVerificationTypeSelection(type) {
        setVerificationType(type);

        if (type === "nn" || type === "nncs" || type === "simulink-system") {
            setCurrentStep(2);
        }
    }


    async function callAPI() {
        if (isVerifying) return;

        setIsVerifying(true);

        try {
            const formData = new FormData();

            formData.append("settings", JSON.stringify({
                ...settings,
                specformat: propertyType
            }));

            formData.append("network_file", networkFile);
            formData.append("property_file", propertyFile);

            const endpoint = verificationType === "nn"
                ? "http://localhost:8000/run-nn-averinn"
                : "http://localhost:8000/run-nncs-averinn";

            if (verificationType === "nncs") {
                formData.append("dynamics_file", dynamicsFile);
            }

            const response = await fetch(endpoint, {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            setBackendStatus(data);
            setCurrentStep(4);

        } catch (error) {
            console.error(error);
        } finally {
            setIsVerifying(false);
        }
    }

    return (
        <slContext.Provider value={contextData}>
            <div className="background">
                <Header />

                <ProgressSteps
                    currentStep={currentStep}
                    setCurrentStep={setCurrentStep}
                />

                {currentStep === 1 && (
                    <div>
                        <h1 className="h3 fw-bold text-center">
                            Select Verification Type
                        </h1>

                        <VerifTypePanel onSelectType={handleVerificationTypeSelection} />
                    </div>
                )}

                {isSimulinkType && currentStep > 1 && (
                    <Simulink />
                )}

                {isAverinnType && currentStep === 2 && (
                    <section className="container py-4">
                        <h1 className="h3 fw-bold">
                            Neural Network Property Checking
                        </h1>

                        <p className="fs-5">
                            Provide the model and property file to continue.
                        </p>

                        <div className="row g-4 mt-3">
                            <div className={verificationType === "nncs" ? "col-12 col-lg-4" : "col-12 col-lg-6"}>
                                <UploadCard
                                    title="Neural Network"
                                    formatLabel="Format"
                                    formatValue={settings.nnFormat}
                                    formatOptions={["ONNX", "SHERLOCK", "ISHERLOCK","NNET"]}
                                    onFormatChange={(newFormat) =>
                                        setSettings((previousSettings) => ({
                                            ...previousSettings,
                                            nnFormat: newFormat,
                                        }))
                                    }
                                    fileLabel="Model file"
                                    file={networkFile}
                                    onFileChange={setNetworkFile}
                                    acceptedFileTypes=".onnx,.nnet,.sherlock,.isherlock"
                                    primaryButtonText="View Network"
                                    secondaryButtonText="Validate Network"
                                />
                            </div>

                            <div className={verificationType === "nncs" ? "col-12 col-lg-4" : "col-12 col-lg-6"}>
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
                                    secondaryButtonText="Validate Property"
                                />
                            </div>

                            {verificationType === "nncs" && (
                                <div className="col-12 col-lg-4">
                                    <UploadCard
                                        title="Dynamics File"
                                        formatLabel="Format"
                                        formatValue="INI"
                                        formatOptions={["INI"]}
                                        onFormatChange={() => {}}
                                        fileLabel="Dynamics file"
                                        file={dynamicsFile}
                                        onFileChange={setDynamicsFile}
                                        acceptedFileTypes=".ini"
                                        primaryButtonText="View Dynamics"
                                        secondaryButtonText="Validate Dynamics"
                                    />
                                </div>
                            )}
                        </div>

                        <hr className="section-divider" />

                        <div className="bottom-panel d-flex justify-content-between">
                            <button
                                onClick={() => setCurrentStep(1)}
                                className="btn btn-primary px-4"
                            >
                                Previous Page
                            </button>

                            <button
                                onClick={() => setCurrentStep(3)}
                                className="btn btn-primary px-4"
                            >
                                Continue to Settings
                            </button>
                        </div>
                    </section>
                )}

                {isAverinnType && currentStep === 3 && (
                    <section className="container py-4">
                        <SettingsSection
                            settings={settings}
                            setSettings={setSettings}
                        />

                        <hr className="section-divider" />

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
                                {isVerifying ? "Running Verification..." : "Start Verification"}
                            </button>
                        </div>
                    </section>
                )}

                {isAverinnType && currentStep === 4 && (
                    <section className="container py-4">
                        <h1 className="h3 fw-bold text-center">
                            Analysis Results
                        </h1>

                        <ResultsPanel backendStatus={backendStatus} />

                        <hr className="section-divider" />

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
        </slContext.Provider>
    );
}

export default App;
