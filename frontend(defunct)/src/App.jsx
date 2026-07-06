import { useState } from "react";
import "./App.css";
import Header from "./Header";
import ProgressSteps from "./ProgressSteps";
import SettingsSection from "./SettingsSection";
import UploadCard from "./UploadCard";
import ResultsPanel from "./ResultsPanel";

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

    const [currentStep, setCurrentStep] = useState(2);
    const [networkFormat, setNetworkFormat] = useState("ONNX");
    const [networkFile, setNetworkFile] = useState(null);

    const [propertyType, setPropertyType] = useState("VNNLIB");
    const [propertyFile, setPropertyFile] = useState(null);

    async function callAPI() {
        const response = await fetch("http://localhost:8000/run-nncs-averinn", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(settings)
        });

        const data = await response.json();
        setBackendStatus(data);
    }

    return (

    
        <div className="background">
            <Header />

            <ProgressSteps
                currentStep={currentStep}
                setCurrentStep={setCurrentStep}
            />

            {currentStep === 1 && (
                <h1 className="generic-h1">Body for first page</h1>
            )}

            {currentStep === 2 && (
                <section className="container py-4">

                    <h1 className="h3 fw-bold">Neural Network Property Checking</h1>

                    <p className="fs-5">
                        Provide the model and property file to continue.
                    </p>

                    <div className="row g-4 mt-3">

                        <div className="col-12 col-lg-6">
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

                        <div className="col-12 col-lg-6">
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

                    </div>

                </section>
            )}

            {currentStep === 3 && (
                <>
                    <SettingsSection
                        settings={settings}
                        setSettings={setSettings}
                    />
                    <hr className="section-divider" />

                    <div className="bottom-panel d-flex justify-content-center">
                        <button onClick={callAPI} className="btn btn-primary px-4">
                            Start Verification
                        </button>
                    </div>

                    <hr className="section-divider" />

                    <ResultsPanel backendStatus={backendStatus} />
                
                </>

            )}
        </div>
    );
}

export default App;