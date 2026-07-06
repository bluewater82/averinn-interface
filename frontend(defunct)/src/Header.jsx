import "./Header.css";

function Header() {
    return (
        <header className="app-header d-flex align-items-center">
            <img src="blacklogo.png" className="app-logo me-3" />

            <div>
                <h1 className="app-title">AVERINN</h1>
                <p className="app-subtitle">sub-title</p>
            </div>
        </header>
    );
}

export default Header;