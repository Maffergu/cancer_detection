import './App.css';
import { useState, useRef, useEffect } from 'react';
import Modal from 'react-modal';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';

Modal.setAppElement('#root');

export default function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = e => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadedFile(file);

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

  
  
  
  const handleProcessClick = () => {
    // Aquí enviar `uploadedFile` al backend cuando esté listo
    console.log('Procesando:', uploadedFile);
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
            <li>Upload image</li>
            <li>Click “process image”</li>
            <li>Get results</li>
          </ol>
        </div>

        <div className="actions">
          <button className="upload-btn" onClick={handleUploadClick}>
            <img src="/images/upload-icon.png" alt="Upload image" />
          </button>
          <input
            type="file"
            accept=".png,.jpg,.jpeg"
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
          <button className="process-btn"
            onClick={handleProcessClick}
            disabled={!uploadedFile}>
            <img src="/images/process-icon.png" alt="Process image" />
            </button>
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
         onClick={() => previewUrl && setIsModalOpen(true)}
         style={{ cursor: previewUrl ? 'zoom-in' : 'default' }}
       >
        {previewUrl && <img src={previewUrl} alt="Original" />}
      </div>
            </div>
            {/* MODAL + ZOOM */}
     {previewUrl && (
       <Modal
         isOpen={isModalOpen}
         onRequestClose={() => setIsModalOpen(false)}
         style={{
           overlay: { backgroundColor: 'rgba(0,0,0,0.75)' },
           content: {
             inset: '5%',
             padding: 0,
             border: 'none',
             background: 'transparent'
           }
         }}
       >
         <TransformWrapper
           initialScale={1}
           minScale={0.5}
           maxScale={5}
           wheel={{ step: 0.2 }}
         >
           {({ zoomIn, zoomOut, resetTransform }) => (
             <div className="zoom-modal">
               <div className="tools">
                 <button onClick={zoomIn}>＋</button>
                 <button onClick={zoomOut}>－</button>
                 <button onClick={resetTransform}>⟳</button>
                 <button onClick={() => setIsModalOpen(false)}>✕</button>
               </div>
               <TransformComponent>
                 <img
                   src={previewUrl}
                   alt="Zoomed"
                   style={{
                     width: '100%',
                     height: '100%',
                     objectFit: 'contain'
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
              <div className="image-box">
                <div className="zoom">…</div>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* FOOTER */}
      <div className="footer-accent">
        <div className="classification">
          CLASIFICACIÓN Y PRECISIÓN:
        </div>
        <div className="resultado">
          "result"
        </div>
      </div>
    </div>
  );
}
