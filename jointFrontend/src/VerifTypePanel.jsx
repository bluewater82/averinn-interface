import NNImage from "./assets/icons/nn.png";
import NNCSImage from "./assets/icons/avhybimg.png";
import Cog from "./assets/icons/avwipimg.png";
import LinearUp from "./assets/icons/avlin.png";
import NonLinear from "./assets/icons/avnlin.png";
import SimLink from "./assets/icons/avssimg.png";
import "./VerifTypePanel.css";

function VerifTypePanel({ onSelectType }) {
    return (
        <section className="container my-5">
            <div className="verification-grid">
                <div className="col-md-4">
                    <button
                        className="verif-type-card"
                        onClick={() => onSelectType("nn")}
                    >
                        <img src={NNImage} className="type-icon" alt="Neural Network" />
                        <span>NN Controller</span>
                    </button>
                </div>

                <div className="col-md-4">
                    <button
                        className="verif-type-card"
                        onClick={() => onSelectType("nncs")}
                    >
                        <img src={NNCSImage} className="type-icon-enlarge" alt="NNCS" />
                        <span>NN Controller</span>
                        <span>With Dynamics</span>
                    </button>
                </div>

                <div className="col-md-4">
                    <button
                        className="verif-type-card"
                        onClick={() => onSelectType("linear-system")}
                    >
                        <img src={LinearUp} className="type-icon" alt="Linear System" />
                        <span>NN-Controlled</span>
                        <span>Linear System</span>
                    </button>
                </div>

                <div className="col-md-4">
                    <button
                        className="verif-type-card"
                        onClick={() => onSelectType("nonlinear-system")}
                    >
                        <img src={NonLinear} className="type-icon" alt="Nonlinear System" />
                        <span>NN-Controlled</span>
                        <span>Nonlinear System</span>
                    </button>
                </div>

                <div className="col-md-4">
                    <button
                        className="verif-type-card"
                        onClick={() => onSelectType("todo-1")}
                    >
                        <img src={SimLink} className="type-icon" alt="TODO 1" />
                        <span>Simulink Systems</span>
                    </button>
                </div>

                <div className="col-md-4">
                    <button
                        className="verif-type-card"
                        onClick={() => onSelectType("todo-2")}
                    >
                        <img src={Cog} className="type-icon" alt="TODO 2" />
                        <span>TODO 2</span>
                    </button>
                </div>
            </div>
        </section>
    );
}

export default VerifTypePanel