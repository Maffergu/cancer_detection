import "./App.css";
import { useState, useRef, useEffect } from "react";
import Modal from "react-modal";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

Modal.setAppElement("#root");

export default function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const fileInputRef = useRef(null);
  const [loadingPreview, setLoadingPreview] = useState(false);

  const [isSRModalOpen, setIsSRModalOpen] = useState(false);
  const [srImageLoaded, setSrImageLoaded] = useState(false);
  const [previewImageLoaded, setPreviewImageLoaded] = useState(false);

  const [origDimensions, setOrigDimensions] = useState({ width: null, height: null });
  const [srDimensions, setSrDimensions] = useState({ width: null, height: null });

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadedFile(file);
    setLoadingPreview(true);

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleImageLoad = () => {
    setTimeout(() => {
      setLoadingPreview(false);
    }, 1000);
  };

  const [resultImage, setResultImage] = useState(null);
  const [classification, setClassification] = useState(null);
  const [processing, setProcessing] = useState(false);

  const handleProcessClick = async () => {
    if (!uploadedFile) return;

    setProcessing(true);

    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
      const response = await fetch("http://localhost:8000/process-image/", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setResultImage("http://localhost:8000" + data.image_url);
        setClassification({
          class: data.class,
          confidence: data.confidence,
        });
      } else {
        alert("Error al procesar la imagen: " + data.error);
      }
    } catch (error) {
      console.error("Error de red:", error);
      alert("Error de red");
    }

    setProcessing(false);
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="app">
      {/* HEADER / BANNER */}
      <div className="banner">
        <div className="logo">
          <h1>AURoRA</h1>
        </div>

        <div className="instructions">
          <ol>
            <li>Upload image (.png / .jpg)</li>
            <li>Click “process image”</li>
            <li>Get results</li>
          </ol>
        </div>

        <div className="actions">
          <div className="btn-with-tooltip">
            <button
              className="upload-btn"
              onClick={handleUploadClick}
              aria-label="Select a .png/.jpg/.jpeg from your device"
            >
              <img src="/images/upload-icon.png" alt="Upload image" />
            </button>
            <span className="tooltip">Upload an image</span>
          </div>
          <input
            type="file"
            accept=".png,.jpg,.jpeg"
            ref={fileInputRef}
            style={{ display: "none" }}
            onChange={handleFileChange}
          />
          <div className="btn-with-tooltip">
            <button
              className="process-btn"
              onClick={handleProcessClick}
              aria-label="Proceess the uploaded image"
              disabled={!uploadedFile}
            >
              <img src="/images/process-icon.png" alt="Process image" />
            </button>
            <span className="tooltip">Process image</span>
          </div>
        </div>
      </div>

      <main className="content">
        <section className="section1">
          <div className="images-row">
            {/* ORIGINAL */}
            <div className="image-container">
              <div className="label">ORIGINAL</div>
              <div
                className="image-box"
                onClick={() =>
                  previewUrl && previewImageLoaded && setIsModalOpen(true)
                }
                style={{
                  cursor:
                    previewUrl && previewImageLoaded ? "zoom-in" : "default",
                }}
              >
                {loadingPreview && <div className="spinner" />}
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt="Original Preview"
                    onLoad={(e) => {
                      handleImageLoad();
                      setPreviewImageLoaded(true);
                      setOrigDimensions({
                        width: e.target.naturalWidth,
                        height: e.target.naturalHeight
                      });
                    }}
                  />
                )}
              </div>
            </div>
            {/* MODAL + ZOOM */}
            {previewUrl && previewImageLoaded && (
              <Modal
                isOpen={isModalOpen}
                onRequestClose={() => setIsModalOpen(false)}
                className="zoom-modal-content"
                overlayClassName="zoom-modal-overlay"
              >
                <TransformWrapper>
                  {(utils) => (
                    <div className="zoom-modal">
                      <div className="tools">
                        <button onClick={() => utils.zoomIn()}>＋</button>
                        <button onClick={() => utils.zoomOut()}>－</button>
                        <button onClick={() => utils.resetTransform()}>
                          ⟳
                        </button>
                        <button onClick={() => setIsModalOpen(false)}>✕</button>
                      </div>
                      <TransformComponent>
                        <img
                          src={previewUrl}
                          alt="Zoomed Original"
                          style={{
                            width: "100%",
                            height: "100%",
                            objectFit: "contain",
                          }}
                        />
                      </TransformComponent>
                    </div>
                  )}
                </TransformWrapper>
              </Modal>
            )}

            {/* SUPER RESOLUCIÓN */}
            <div className="image-container">
              <div className="label">SUPER RESOLUCIÓN</div>
              <div
                className="image-box"
                onClick={() => resultImage && setIsSRModalOpen(true)}
                style={{ cursor: resultImage ? "zoom-in" : "default" }}
              >
                {processing && <div className="spinner" />}
                {!processing && resultImage && (
                  <img
                    src={resultImage}
                    alt="Resultado SR"
                    onLoad={(e) => {
                      handleImageLoad();
                      setSrImageLoaded(true);
                      setSrDimensions({
                        width: e.target.naturalWidth,
                        height: e.target.naturalHeight
                      });
                    }}
                  />
                )}
              </div>
            </div>

            {/* MODAL + ZOOM para SUPER RESOLUCIÓN */}
            {resultImage && srImageLoaded && origDimensions.width && srDimensions.width && (
              <Modal
                isOpen={isSRModalOpen}
                onRequestClose={() => setIsSRModalOpen(false)}
                className="zoom-modal-content"
                overlayClassName="zoom-modal-overlay"
              >
                <TransformWrapper
                  initialScale={origDimensions.width && srDimensions.width ? origDimensions.width / srDimensions.width : 1}
                  minScale={origDimensions.width && srDimensions.width ? origDimensions.width / srDimensions.width : 1}
                  centerOnInit={true}
                  wheel={{ disabled: false }}
                >
                  {({ zoomIn, zoomOut, resetTransform }) => (
                    <div className="zoom-modal">
                      <div className="tools">
                        <button onClick={() => zoomIn()}>＋</button>
                        <button onClick={() => zoomOut()}>－</button>
                        <button onClick={() => resetTransform()}>⟳</button>
                        <button onClick={() => setIsSRModalOpen(false)}>
                          ✕
                        </button>
                      </div>
                      <TransformComponent>
                        <img
                          src={resultImage}
                          alt="Resultado SR Ampliado"
                          style={{
                            width: "100%",
                            height: "100%",
                            objectFit: "contain",
                          }}
                        />
                      </TransformComponent>
                    </div>
                  )}
                </TransformWrapper>
              </Modal>
            )}
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <div className="footer-accent">
        <div className="classification">CLASIFICACIÓN Y PRECISIÓN:</div>
        <div className="resultado">
          {classification ? (
            <>
              <strong>{classification.class}</strong>
              <br />
              Precisión: {(classification.confidence * 100).toFixed(2)}%
            </>
          ) : (
            "Sin resultados"
          )}
        </div>
      </div>
    </div>
  );
}
