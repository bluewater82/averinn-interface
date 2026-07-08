import "./SettingsCards.css";

/**********************************************************************
 * ProblemMethodCard.jsx
 * 
 * Component for the Problem&Method settings
 * 
 * A small group of <select> and <input> that allow the user to define
 * custom settings for their network verification, such as solver, last-layer
 * ReLU, and encoding. User selections are sent to App.jsx to be used to 
 * generate a custom configuration file to be run with the tool.
 * 
 ***********************************************************************/

function ProblemMethodCard({settings, setSettings}) {
    return (
        <div className="verification-card">
            <h2 className="verification-card-title">Problem & Method</h2>

            <div className="form-row-custom">
                <label>Problem type</label>
                <select 
                    className="form-select"
                    value={settings.probType}
                    onChange={(e) =>
                        setSettings({
                            ...settings,
                            probType: e.target.value
                        })
                    }
                >
                    <option>SAFETY</option>
                    <option>REACHABILITY</option>
                </select>
            </div>

            <div className="form-row-custom">
                <label>Technique</label>
                <select 
                    className="form-select"
                    value={settings.technique}
                    onChange={(e) =>
                        setSettings({
                            ...settings,
                            technique: e.target.value
                        })
                    }
                >
                    <option>Propagation</option>
                    <option>MILP</option>
                </select>
            </div>

            <div className="form-row-custom">
                <label>Solver</label>
                <select 
                    className="form-select"
                    value={settings.solverType}
                    onChange={(e) =>
                        setSettings({
                            ...settings,
                            solverType: e.target.value
                        })
                    }
                >
                    <option>Gurobi</option>
                    <option>CPLEX</option>
                </select>
            </div>

            <div className="form-row-custom">
                <label>Last layer ReLU</label>
                <select 
                    className="form-select"
                    value={settings.lastRelu}
                    onChange={(e) =>
                        setSettings({
                            ...settings,
                            lastRelu: e.target.value
                        })
                    }
                >
                    <option>No</option>
                    <option>Yes</option>
                </select>
            </div>

            <div className="form-row-custom">
                <label>Number of loops</label>
                <input 
                    className="form-control" type="number" min="1" placeholder="1" 
                    value={settings.K}
                    onChange={(e) =>
                        setSettings({
                            ...settings,
                            K: Number(e.target.value)
                        })
                    }
                />
            </div>
        </div>
    );
}

export default ProblemMethodCard;