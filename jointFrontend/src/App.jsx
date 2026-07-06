import { useState } from "react";
import "./App.css";
import Header from "./Header";
import ProgressSteps from "./ProgressSteps";
import SettingsSection from "./SettingsSection";
import UploadCard from "./UploadCard";
import ResultsPanel from "./ResultsPanel";
// import UploadSlx from "./UploadSlx.jsx"
import NavButtonPanel from "./NavButtonPanel.jsx"
import ControlTypeCard from "./ControlTypeCard.jsx";

/*************************************************************
 * App.jsx
 * 
 * Main application component for the AVERINN frontend.
 * 
 * Responsibilities:
 *  - Maintains global application state
 *  - Controls navigation between verification stages
 *  - Coordinates communication with FastAPI backend
 *  - Passes shared state to child components
 * 
 * Workflow:
 *  - Header->Progress Bar->Current Verification Page->Results
 * 
 *************************************************************/



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

    const [currentNetType, setNetType] = useState("Dynamic");

    const [dynamicsFile, setDynamicsFile] = useState(null);
    const [currentType, setCurrentType] = useState(0); // Verification type
    const [currentStep, setCurrentStep] = useState(2);
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
    }

    return (

    
        <div className="background">
            <Header />

            <ProgressSteps
                currentStep={currentStep}
                setCurrentStep={setCurrentStep}
            />

            {currentType === 0 &&
                (currentStep === 1 && (
                    <h1 className="generic-h1">Body for first page</h1>
                ) ) ||

                (currentStep === 2 && (
                    <section className="container py-4">

                        <h1 className="h3 fw-bold">Neural Network Property Checking</h1>

                        <p className="fs-5">
                            Provide the model and property file to continue.
                        </p>

                        
                        
                        <div className="row g-4 mt-3">

                            <div className="col-lg-4">
                                <ControlTypeCard 
                                    currentNetType={currentNetType}
                                    setNetType={setNetType}
                                />
                            </div>

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

                            {currentNetType === "Dynamic" && (
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

                            <hr className="section-divider" />

                            <div className="bottom-panel d-flex justify-content-between">

                                <button onClick={() => setCurrentStep(1)} className="btn btn-primary px-4">
                                    Previous Page
                                </button>
                                <button onClick={() => setCurrentStep(3)} className="btn btn-primary px-4">
                                    Continue to Settings
                                </button>
                            </div>

                        </div>

                    </section>
                )) ||
                
                (currentStep === 3 && (
                    <>
                        <SettingsSection
                            settings={settings}
                            setSettings={setSettings}
                        />
                        <hr className="section-divider" />

                        <div className="bottom-panel d-flex justify-content-between">

                            <button onClick={() => setCurrentStep(2)} className="btn btn-primary px-4">
                                Previous Page
                            </button>
                            <button onClick={callAPI} className="btn btn-primary px-4">
                                Start Verification
                            </button>
                        </div>

                        

                        <hr className="section-divider" />

                        <ResultsPanel backendStatus={backendStatus} />
                    
                    </>

                ))
            }
            { currentType === 4 &&
                (currentStep === 0 && (
                    <UploadSlx 
                        part={currentStep} 
                        setPart={setCurrentStep}
                    />
                    )
                    
                )
            }
        </div>
    );
}

export default App;