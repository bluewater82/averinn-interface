import "./SettingsCards.css";

function SelectedInputsCard() {
    return (
        <div className="verification-card">
            <h2 className="verification-card-title">Selected Inputs</h2>

            <SelectedInput label="Verification type" value="NN property checking" />
            <SelectedInput label="Model format" value="ONNX" />
            <SelectedInput label="Model file" value="controller.onnx" />
            <SelectedInput label="Property file" value="spec.vnnlib" />
            <SelectedInput label="Expected output" value="SAT/UNSAT/UNKNOWN" />
        </div>
    );
}

function SelectedInput({ label, value }) {
    return (
        <div className="selected-input-row">
            <span>{label}</span>
            <span>...........</span>
            <span>{value}</span>
        </div>
    );
}

export default SelectedInputsCard;