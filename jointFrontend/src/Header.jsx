import "./Header.css";

/****************************************************************
 * Header.jsx
 * 
 * Component for the main header.
 * 
 * Structure for logo, tile, subtitle, and help icon
 * 
 ****************************************************************/

function Header() {
    return (
        <header className="app-header d-flex align-items-center">
            <img src="blacklogo.png" className="app-logo me-3" />

            <div>
                <h1 className="app-title">AVERINN</h1>
                <p className="app-subtitle">sub-title</p>
            </div>

            <button className="help-button ms-auto">
                ? Help
            </button>
        </header>
    );
}

export default Header;