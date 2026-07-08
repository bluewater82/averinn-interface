import "./SettingsCards.css";

/**********************************************************************
 * AbstractionCard.jsx
 * 
 * Component for the abstraction settings
 * 
 * A small group of <select> and <input> that allow the user to define
 * if they want to use abstract neurons for their verification and how
 * many they want to use. Selections are sent and stored in App.jsx for
 * use in generating custom configuration file used with the AVERINN tool.
 * 
 ***********************************************************************/

function AbstractionCard({settings, setSettings}) {
    return (
        <div className="verification-card">
            <h2 className="verification-card-title">Abstraction</h2>

            <div className="form-row-custom">
                <label>Use abstraction</label>
                <select 
                    className="form-select"
                    value={settings.absRequired}
                    onChange={(e) =>
                        setSettings({
                            ...settings,
                            absRequired: e.target.value
                        })
                    }
                >
                    <option>No</option>
                    <option>Yes</option>
                </select>
            </div>

            <div className="form-row-custom">
                <label>Number of abstract nodes</label>
                <input 
                    className="form-control" type="number" min="1" placeholder="1"
                    value= {settings.numOfAbsNodes}
                    onChange={(e) =>
                        setSettings({
                            ...settings,
                            numOfAbsNodes: Number(e.target.value)
                        })
                    }
                />
            </div>

            <div className="form-row-custom">
                <label>Abstraction type</label>
                <select 
                    className="form-select"
                    value={settings.absType}
                    onChange={(e) =>
                        setSettings({
                            ...settings,
                            absType: e.target.value
                        })
                    }
                >
                    <option>INTERVAL</option>
                    <option>STAR</option>
                </select>
            </div>

            <div className="form-row-custom">
                <label>Partition type</label>
                <select 
                className="form-select"
                value={settings.partitionType}
                onChange={(e) =>
                    setSettings({
                        ...settings,
                        partitionType: e.target.value
                    })
                }
            >
                    <option>FIXED</option>
                    <option>STAR</option>
                </select>
            </div>
        </div>
    );
}

export default AbstractionCard;