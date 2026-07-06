import "./ProgressSteps.css";

/**************************************************************************
 * ProgressSteps.jsx
 * 
 * Component that handles the icons that inform the user of what step
 * they are currently in at any given point during the configuration
 * of the tool.
 * 
 * Progression:
 *  - 1)Select Type->2)Files & Property->3)Verification Settings->4)Results
 * 
 * Tied directly to state for telling App.jsx which components to display
 * 
 **************************************************************************/

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