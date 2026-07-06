import ProblemMethodCard from "./ProblemMethodCard";
import AbstractionCard from "./AbstractionCard";
import SelectedInputsCard from "./SelectedInputsCard";

function SettingsSection({settings, setSettings}) {
    return (
        <section className="container py-4">
            <h1 className="h3 fw-bold">Neural Network Property Checking</h1>
            <p className="fs-5">
                Configure the analysis options for the selected model and property.
            </p>

            <div className="row g-4 mt-3">
                <div className="col-12 col-lg-4">
                    <ProblemMethodCard 
                    settings={settings}
                    setSettings={setSettings}
                    />
                </div>

                <div className="col-12 col-lg-4">
                    <AbstractionCard 
                    settings={settings}
                    setSettings={setSettings}
                    />
                </div>

                <div className="col-12 col-lg-4">
                    <SelectedInputsCard />
                </div>
            </div>
        </section>
    );
}

export default SettingsSection;