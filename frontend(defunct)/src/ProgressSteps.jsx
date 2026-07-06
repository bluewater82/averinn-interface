import "./ProgressSteps.css";

function ProgressSteps({ currentStep, setCurrentStep }) {
    const steps = [
        "Select Type",
        "Files & Property",
        "Verification Settings",
        "Results"
    ];

    return (
        <div className="progress-steps">
            {steps.map((step, index) => {
                const stepNumber = index + 1;
                const isComplete = stepNumber < currentStep;
                const isActive = stepNumber === currentStep;

                return (
                    <div className="progress-step-group" key={step}>
                        <button
                            className={`progress-step ${
                                isComplete ? "complete" : ""
                            } ${isActive ? "active" : ""}`}
                            onClick={() => setCurrentStep(stepNumber)}
                        >
                            <span className="step-circle">
                                {isComplete ? "✓" : stepNumber}
                            </span>

                            <span className="step-label">{step}</span>
                        </button>

                        {stepNumber < steps.length && (
                            <div
                                className={`step-line ${
                                    isComplete ? "complete-line" : ""
                                }`}
                            />
                        )}
                    </div>
                );
            })}
        </div>
    );
}

export default ProgressSteps;