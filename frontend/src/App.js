import './App.css';

export default function App() {
  return (
    <div className="app">
      {/* HEADER / BANNER */}
      <div className="banner">
        <div className="logo">
          <h1>AURoRA</h1>
        </div>

        <div className="instructions">
          <ol>
            <li>Upload image</li>
            <li>Click “process image”</li>
            <li>Get results</li>
          </ol>
        </div>

        <div className="actions">
          <button className="upload-btn">📤</button>
          <button className="process-btn">PROCESS IMAGE</button>
        </div>
      </div>

      <main>
        {/* SECCIÓN 1 */}
        <section className="section1">

          <div className="images-row">
            {/* Original */}
            <div className="image-container">
              <div className="label">ORIGINAL</div>
              <div className="image-box">
                <img src="/images/dummyPic.jpg" alt="Original" />
              </div>
            </div>

            {/* Super Resolución */}
            <div className="image-container">
              <div className="label">SUPER RESOLUCIÓN</div>
              <div className="image-box">
                {/* Aquí entra la imagen procesada */}
                <div className="zoom">
                  <span>🔍＋</span>
                  <span>🔍−</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* CLASIFICACIÓN */}
        <section className="classification">
          CLASIFICACIÓN Y PRECISIÓN:
          {/* Aquí pondrías tus resultados */}
        </section>
      </main>

      {/* PIE DE PÁGINA ACENTUADO */}
      <div className="footer-accent"></div>
    </div>
  );
}
