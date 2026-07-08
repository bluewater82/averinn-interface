import { useState } from "react";
import "./App.css";
import Header from "./Header";
import ProgressSteps from "./ProgressSteps";
import SettingsSection from "./SettingsSection";
import UploadCard from "./UploadCard";
import ResultsPanel from "./ResultsPanel";
import VerifTypePanel from "./VerifTypePanel.jsx";

function App() {
    const [backendStatus, setBackendStatus] = useState(null);

    const [settings, setSettings] = useState({
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
    const [currentType, setCurrentType] = useState(0);
    const [currentStep, setCurrentStep] = useState(1);

    const [networkFormat, setNetworkFormat] = useState("ONNX");
    const [networkFile, setNetworkFile] = useState(null);

    const [propertyType, setPropertyType] = useState("VNNLIB");
    const [propertyFile, setPropertyFile] = useState(null);

    async function callAPI() {
        const formData = new FormData();

        formData.append("settings", JSON.stringify({
            ...settings,
            nnformat: networkFormat,
            specformat: propertyType
        }));

        formData.append("network_file", networkFile);
        formData.append("property_file", propertyFile);
        formData.append("dynamics_file", dynamicsFile);

        const response = await fetch("http://localhost:8000/run-nncs-averinn", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        setBackendStatus(data);
        setCurrentStep(4);
    }

    return (
        <div className="background">
            <Header />

            <ProgressSteps
                currentStep={currentStep}
                setCurrentStep={setCurrentStep}
            />

            {currentType === 0 && (
                <>
                    {currentStep === 1 && (
                        <div>
                            <h1 className="h3 fw-bold text-center">
                                Select Verification Type
                            </h1>

                            <VerifTypePanel />

                            <hr className="section-divider" />

                            <div className="bottom-panel d-flex justify-content-center">
                                <button
                                    onClick={() => setCurrentStep(2)}
                                    className="btn btn-primary px-4"
                                >
                                    Upload Files
                                </button>
                            </div>
                        </div>
                    )}

                    {currentStep === 2 && (
                        <section className="container py-4">
                            <h1 className="h3 fw-bold">
                                Neural Network Property Checking
                            </h1>

                            <p className="fs-5">
                                Provide the model and property file to continue.
                            </p>

                            <div className="row g-4 mt-3">
                                <div className="col-12 col-lg-4">
                                    <UploadCard
                                        title="Neural Network"
                                        formatLabel="Format"
                                        formatValue={networkFormat}
                                        formatOptions={["ONNX", "SHERLOCK", "NNET"]}
                                        onFormatChange={setNetworkFormat}
                                        fileLabel="Model file"
                                        file={networkFile}
                                        onFileChange={setNetworkFile}
                                        acceptedFileTypes=".onnx,.nnet,.sherlock"
                                        primaryButtonText="Preview Network"
                                        secondaryButtonText="Validate Network"
                                    />
                                </div>

                                <div className="col-12 col-lg-4">
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

                    {currentStep === 3 && (
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
                                    className="btn btn-primary px-4"
                                >
                                    Start Verification
                                </button>
                            </div>
                        </section>
                    )}

                    {currentStep === 4 && (
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
                </>
            )}
        </div>
    );
}

export default App;