

function NavButtonPanel({ currentStep, setCurrentStep, onStartVerification }) {
    function goPrevious() {
        setCurrentStep(currentStep - 1);
    }

    function goNext() {
        setCurrentStep(currentStep + 1);
    }

    const isFirstStep = currentStep === 1;
    const isSettingsStep = currentStep === 3;

    return (
        <>
            <hr className="section-divider" />

            <div className="bottom-panel d-flex justify-content-between">
                <button
                    onClick={goPrevious}
                    className="btn btn-primary px-4"
                    disabled={isFirstStep}
                >
                    Previous Page
                </button>

                {isSettingsStep ? (
                    <button
                        onClick={onStartVerification}
                        className="btn btn-primary px-4"
                    >
                        Start Verification
                    </button>
                ) : (
                    <button
                        onClick={goNext}
                        className="btn btn-primary px-4"
                    >
                        Continue to Settings
                    </button>
                )}
            </div>
        </>
    );
}

export default NavButtonPanel;