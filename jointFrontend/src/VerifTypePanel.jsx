import { ChartLine } from "lucide-react";
import { ChartSplineIcon } from "lucide-react";
import { SettingsIcon } from "lucide-react";
import { Waypoints } from "lucide-react";
import { Network } from "lucide-react";
import { Workflow } from "lucide-react";
import NNImage from "./assets/icons/mw-neural-net.png";
import NNCSImage from "./assets/icons/mw-system-diagram.png";
import Cog from "./assets/icons/fa-cog.png";
import LinearUp from "./assets/icons/linear.png";
import NonLinear from "./assets/icons/mat-trending-up.png";
import "./VerifTypePanel.css";

function VerifTypePanel() {
    return (
        <section className="container my-5">
            <div className="verification-grid">
                <div className="col-md-4">
                    <button className="verif-type-card">
                        <img src={NNImage}
                                className="type-icon"
                                alt="Neural Network"
                        />
                        
                        <span>NN Controller</span>
                        
                    </button>
                </div>
                <div className="col-md-4">
                    <button className="verif-type-card">
                        <div className="icon-stack">
                            <img src={NNImage}
                                className="main-icon"
                                alt="Neural Network" />
                            <img src={NNCSImage}
                                className="overlay-icon"
                                alt="NNCS" />
                        </div>
                        
                        <span>NN Controller</span>
                        <span>With Dynamics</span>
                        
                    </button>
                </div>
                <div className="col-md-4">
                    <button className="verif-type-card">
                        <img src={LinearUp}
                                className="type-icon"
                                alt="Neural Network"
                        />
                        
                        <span>NN-Controlled</span>
                        <span>Linear System</span>
                        
                    </button>
                    
                </div>
            
                <div className="col-md-4">
                    <button className="verif-type-card">
                        <img src={NonLinear}
                                className="type-icon"
                                alt="Neural Network"
                        />
                        
                        <span>NN-Controlled</span>
                        <span>Nonlinear System</span>
                        
                    </button>
                </div>
                <div className="col-md-4">
                    <button className="verif-type-card">
                        <img src={Cog}
                                className="type-icon"
                                alt="Neural Network"
                        />
                        
                        <span>TODO 1</span>
                        
                    </button>
                </div>
                <div className="col-md-4">
                    <button className="verif-type-card">
                        <img src={Cog}
                                className="type-icon"
                                alt="Neural Network"
                        />
                        
                        <span>TODO 2</span>
                        
                    </button>
                </div>
            </div>

        </section>
    )
}

export default VerifTypePanel