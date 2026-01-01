import { useEffect, useState, useRef } from "react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import axios from "axios";
import "./App.scss";
import {
  FiArrowRight,
  FiX,
  FiClock,
  FiUsers,
  FiSun,
  FiMoon,
  FiCamera,
  FiImage,
} from "react-icons/fi";
import {
  LuChefHat,
  LuUtensils,
  LuScanLine,
  LuRefrigerator,
} from "react-icons/lu";

gsap.registerPlugin(ScrollTrigger);

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null); // Video ID state
  const [searchAnalysis, setSearchAnalysis] = useState(null);
  const [showAbout, setShowAbout] = useState(false); // New About Modal state
  const [theme, setTheme] = useState("light");

  // Camera State
  const [showCamera, setShowCamera] = useState(false);
  const [cameraMode, setCameraMode] = useState("pantry"); // 'pantry' | 'barcode'
  const [processingImage, setProcessingImage] = useState(false);
  const [fridgeImage, setFridgeImage] = useState(null); // Store user's fridge photo
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  const cursorRef = useRef(null);

  const quickPicks = [
    {
      name: "Italian Night",
      ingredients: "pasta, tomato, basil, garlic, olive oil",
    },
    {
      name: "Healthy Green",
      ingredients: "spinach, kale, avocado, lemon, chicken",
    },
    { name: "Comfort Food", ingredients: "potato, cheese, bacon, cream" },
    {
      name: "Indian Spice",
      ingredients: "paneer, tomato, onion, garam masala, rice",
    },
    {
      name: "Mexican Fiesta",
      ingredients: "beans, corn, avocado, lime, cilantro",
    },
  ];

  useEffect(() => {
    const tl = gsap.timeline();
    tl.fromTo(
      ".hero-word",
      { y: 100, opacity: 0, rotate: 5 },
      {
        y: 0,
        opacity: 1,
        rotate: 0,
        duration: 1,
        stagger: 0.1,
        ease: "power4.out",
      }
    )
      .fromTo(
        ".search-container-hero",
        { y: 30, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.8, ease: "power3.out" },
        "-=0.5"
      )
      .fromTo(
        ".quick-picks",
        { opacity: 0 },
        { opacity: 1, duration: 0.8 },
        "-=0.4"
      );
  }, []);

  useEffect(() => {
    const moveCursor = (e) => {
      gsap.to(cursorRef.current, {
        x: e.clientX,
        y: e.clientY,
        duration: 0.1,
        ease: "power2.out",
      });
    };
    window.addEventListener("mousemove", moveCursor);
    return () => window.removeEventListener("mousemove", moveCursor);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  const toggleTheme = () =>
    setTheme((prev) => (prev === "light" ? "dark" : "light"));

  const handleSearch = async (e, overrideQuery = null) => {
    if (e) e.preventDefault();
    const searchQuery = overrideQuery || query;
    if (!searchQuery.trim()) return;

    if (overrideQuery) setQuery(overrideQuery);

    // REMOVED: setFridgeImage(null) - Keep the image if it exists from a scan!

    setLoading(true);

    try {
      const response = await axios.post("http://localhost:8000/recommend", {
        ingredients: searchQuery,
      });

      // Handle new response structure
      if (response.data.results) {
        setResults(response.data.results);
        setSearchAnalysis(response.data.analysis);
      } else {
        // Fallback for old API if needed (though we just changed it)
        setResults(response.data);
      }

      setSearched(true);

      setTimeout(() => {
        const resultsSection = document.getElementById("results");
        resultsSection?.scrollIntoView({ behavior: "smooth" });
      }, 100);

      setTimeout(() => {
        gsap.fromTo(
          ".recipe-card",
          { y: 100, opacity: 0, scale: 0.95 },
          {
            y: 0,
            opacity: 1,
            scale: 1,
            duration: 0.8,
            stagger: 0.1,
            ease: "power4.out",
            clearProps: "all",
          }
        );
      }, 300);
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  // Camera Logic
  const startCamera = async () => {
    setShowCamera(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
      });
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch (err) {
      console.error("Camera access denied:", err);
      // We allow opening the modal even if camera fails, so user can upload
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
    }
    setShowCamera(false);
    setProcessingImage(false);
  };

  const processScanResult = (response) => {
    if (cameraMode === "pantry") {
      const ingredients = response.data.ingredients;
      if (ingredients && ingredients.length > 0) {
        const newIngredients = ingredients.join(", ");
        setQuery((prev) =>
          prev ? `${prev}, ${newIngredients}` : newIngredients
        );
      } else {
        alert("No food identified. Try again.");
      }
    } else {
      // Barcode scan: Clear any previous fridge image
      setFridgeImage(null);

      const product = response.data.product;
      if (product) {
        setQuery((prev) => (prev ? `${prev}, ${product}` : product));
      } else {
        alert("No barcode detected.");
      }
    }
    stopCamera();
  };

  const captureImage = async () => {
    if (!videoRef.current) return;

    setProcessingImage(true);
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
    const imageBase64 = canvas.toDataURL("image/jpeg");

    // Save image if in pantry mode
    if (cameraMode === "pantry") {
      setFridgeImage(imageBase64);
    }

    try {
      let endpoint = cameraMode === "pantry" ? "/scan/pantry" : "/scan/barcode";
      const response = await axios.post(`http://localhost:8000${endpoint}`, {
        image: imageBase64,
      });
      processScanResult(response);
    } catch (error) {
      console.error("Scan Error:", error);
      alert("Scanning failed. Please try again.");
    } finally {
      setProcessingImage(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setProcessingImage(true);
    const reader = new FileReader();

    reader.onloadend = async () => {
      const imageBase64 = reader.result;

      // Save image if in pantry mode
      if (cameraMode === "pantry") {
        setFridgeImage(imageBase64);
      }

      try {
        let endpoint =
          cameraMode === "pantry" ? "/scan/pantry" : "/scan/barcode";
        const response = await axios.post(`http://localhost:8000${endpoint}`, {
          image: imageBase64,
        });
        processScanResult(response);
      } catch (error) {
        console.error("Upload Error:", error);
        alert("Processing failed. Please try again.");
      } finally {
        setProcessingImage(false);
      }
    };

    reader.readAsDataURL(file);
  };

  const openRecipe = async (recipe) => {
    setSelectedRecipe(recipe);
    setVideoUrl(null); // Clear previous

    try {
      const response = await axios.get(
        `http://localhost:8000/recipe/${recipe.id}/video`
      );
      if (response.data.videoId) {
        setVideoUrl(response.data.videoId);
      }
    } catch (error) {
      console.error("Failed to fetch video:", error);
    }
  };

  const closeRecipe = () => {
    setSelectedRecipe(null);
    setVideoUrl(null);
  };

  return (
    <div className="app-container">
      <div className="custom-cursor" ref={cursorRef}></div>

      <nav className="nav">
        <div className="logo">KENSHOKU AI</div>
        <div className="nav-controls">
          <button className="about-btn" onClick={() => setShowAbout(true)}>
            HOW IT WORKS
          </button>
          <button className="theme-toggle" onClick={toggleTheme}>
            {theme === "light" ? <FiMoon /> : <FiSun />}
          </button>
        </div>
      </nav>

      <section className="hero">
        <div className="hero-content">
          <div className="hero-title-wrapper">
            <h1 className="hero-title">
              <span className="hero-word">WHAT'S</span>
              <span className="hero-word">IN</span>
              <span className="hero-word">YOUR</span>
              <br className="mobile-break" />
              <span className="hero-word highlight">FRIDGE?</span>
            </h1>
          </div>

          <div className="search-container-hero">
            <form onSubmit={handleSearch} className="search-form-hero">
              <input
                type="text"
                placeholder="chicken, garlic, lemon..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="hero-input"
              />
              <div className="action-buttons">
                <button
                  type="button"
                  className="camera-btn"
                  onClick={startCamera}
                >
                  <FiCamera />
                </button>
                <button type="submit" className="hero-submit-btn">
                  {loading ? (
                    <span className="loader"></span>
                  ) : (
                    <FiArrowRight />
                  )}
                </button>
              </div>
            </form>
          </div>

          <div className="quick-picks">
            <p className="quick-label">OR TRY THESE:</p>
            <div className="tags-wrapper">
              {quickPicks.map((pick, i) => (
                <button
                  key={i}
                  className="quick-tag"
                  onClick={() => handleSearch(null, pick.ingredients)}
                >
                  {pick.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      {searched && (
        <section className="results-section" id="results">
          <div className="results-header">
            <h2>CURATED SELECTION</h2>
            <div className="line"></div>
          </div>

          {/* Analysis Banner for Unknown Ingredients */}
          {searchAnalysis &&
            searchAnalysis.unknown_ingredients &&
            searchAnalysis.unknown_ingredients.length > 0 && (
              <div className="analysis-banner">
                <p>
                  ⚠️ <strong>Note:</strong> We couldn't recognize
                  <span className="unknown-words">
                    {" "}
                    "{searchAnalysis.unknown_ingredients.join('", "')}"
                  </span>
                  . Results might be broader than expected.
                </p>
              </div>
            )}

          <div className="recipes-grid">
            {results.map((recipe, index) => (
              <div
                key={recipe.id}
                className={`recipe-card color-variant-${index % 3}`}
                onClick={() => openRecipe(recipe)}
              >
                <div className="card-image-placeholder">
                  {fridgeImage ? (
                    <img
                      src={fridgeImage}
                      alt="Your Fridge"
                      className="card-user-image"
                    />
                  ) : (
                    <LuUtensils />
                  )}
                </div>
                <div className="card-content">
                  <div className="card-meta">
                    <span className="card-number">0{index + 1}</span>
                    <span className="card-score">MATCH</span>
                  </div>
                  <h3>{recipe.title}</h3>
                  <p className="ingredients-preview">
                    {recipe.ingredients.substring(0, 80)}...
                  </p>
                  <div className="card-action">
                    <span>VIEW RECIPE</span>
                    <FiArrowRight />
                  </div>
                </div>
              </div>
            ))}

            {results.length === 0 && !loading && (
              <div className="no-results">
                <p>
                  No matches found. Try simplifying your search (e.g. just
                  "potato").
                </p>
              </div>
            )}
          </div>
        </section>
      )}

      {searched && (
        <footer className="footer">
          <p>© 2025 KENSHOKU AI — POWERED BY NLP</p>
        </footer>
      )}

      {selectedRecipe && (
        <div className="modal-overlay" onClick={closeRecipe}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-btn" onClick={closeRecipe}>
              <FiX />
            </button>

            <div className="modal-header">
              <h2>{selectedRecipe.title}</h2>
              <div className="modal-meta">
                <div className="meta-item">
                  <FiClock /> <span>30m</span>
                </div>
                <div className="meta-item">
                  <FiUsers /> <span>4 servings</span>
                </div>
                <div className="meta-item">
                  <LuChefHat /> <span>Easy</span>
                </div>
              </div>
            </div>

            <div className="modal-body">
              {videoUrl && (
                <div className="modal-section video-section">
                  <h3>WATCH TUTORIAL</h3>
                  <div className="video-wrapper">
                    <iframe
                      src={`https://www.youtube.com/embed/${videoUrl}`}
                      title="YouTube video player"
                      frameBorder="0"
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                      allowFullScreen
                    ></iframe>
                  </div>
                </div>
              )}

              <div className="modal-section">
                <h3>INGREDIENTS</h3>
                <p className="ingredients-text">{selectedRecipe.ingredients}</p>
              </div>

              <div className="modal-section">
                <h3>INSTRUCTIONS</h3>
                <div className="instructions-text">
                  {selectedRecipe.instructions}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ABOUT MODAL */}
      {showAbout && (
        <div className="modal-overlay" onClick={() => setShowAbout(false)}>
          <div
            className="modal-content about-content"
            onClick={(e) => e.stopPropagation()}
          >
            <button className="close-btn" onClick={() => setShowAbout(false)}>
              <FiX />
            </button>

            <div className="modal-header">
              <h2>THE TECH</h2>
              <div className="modal-meta">
                <div className="meta-item">
                  <span>NLP</span>
                </div>
                <div className="meta-item">
                  <span>COMPUTER VISION</span>
                </div>
                <div className="meta-item">
                  <span>GENERATIVE AI</span>
                </div>
              </div>
            </div>

            <div className="modal-body">
              <div className="modal-section">
                <h3>THE INTELLIGENCE</h3>
                <p className="ingredients-text">
                  Kenshoku AI isn't just a keyword matcher. It uses a{" "}
                  <strong>Word2Vec</strong> machine learning model trained on
                  over 100,000 recipes.
                </p>
                <br />
                <p className="ingredients-text">
                  It understands that <em>"cilantro"</em> is semantically
                  similar to <em>"coriander"</em>, and that <em>"butter"</em>{" "}
                  often goes with <em>"garlic"</em>. This allows us to recommend
                  recipes that conceptually fit your ingredients, even if they
                  aren't an exact match.
                </p>
              </div>

              <div className="modal-section">
                <h3>THE VISION</h3>
                <p className="ingredients-text">
                  Our Fridge Scan is powered by{" "}
                  <strong>Llama 3.2 Vision</strong> running on{" "}
                  <strong>Groq's LPU™ Inference Engine</strong>.
                </p>
                <br />
                <p className="ingredients-text">
                  Unlike traditional cloud AI that takes seconds, our vision
                  pipeline analyzes your photo in milliseconds, identifying
                  multiple ingredients instantly and feeding them directly into
                  our recommendation engine.
                </p>
              </div>

              <div className="modal-section">
                <h3>THE STACK</h3>
                <p
                  className="ingredients-text"
                  style={{ fontSize: "1rem", opacity: 0.8 }}
                >
                  Built with React + Vite + GSAP for smooth interactions. <br />
                  Backend powered by FastAPI & Python (NLTK/Gensim).
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* CAMERA MODAL */}
      {showCamera && (
        <div className="camera-overlay">
          <div className="camera-modal">
            <button className="close-camera" onClick={stopCamera}>
              <FiX />
            </button>
            <div className="camera-viewfinder">
              <video ref={videoRef} autoPlay playsInline muted></video>
              {cameraMode === "barcode" && (
                <div className="barcode-guide"></div>
              )}
            </div>

            <div className="camera-controls">
              <div className="mode-switch">
                <button
                  className={`mode-btn ${
                    cameraMode === "pantry" ? "active" : ""
                  }`}
                  onClick={() => setCameraMode("pantry")}
                >
                  <LuRefrigerator /> FRIDGE SCAN
                </button>
                <button
                  className={`mode-btn ${
                    cameraMode === "barcode" ? "active" : ""
                  }`}
                  onClick={() => setCameraMode("barcode")}
                >
                  <LuScanLine /> BARCODE
                </button>
              </div>

              <div className="shutter-row">
                {/* Hidden File Input */}
                <input
                  type="file"
                  accept="image/*"
                  ref={fileInputRef}
                  style={{ display: "none" }}
                  onChange={handleFileUpload}
                />

                {/* Gallery Button */}
                <button
                  className="gallery-btn"
                  onClick={() => fileInputRef.current.click()}
                  disabled={processingImage}
                >
                  <FiImage />
                </button>

                <button
                  className="shutter-btn"
                  onClick={captureImage}
                  disabled={processingImage}
                >
                  {processingImage ? (
                    <div className="loader small"></div>
                  ) : (
                    <div className="inner-circle"></div>
                  )}
                </button>

                {/* Spacer to keep shutter centered */}
                <div className="gallery-spacer"></div>
              </div>

              <p className="camera-hint">
                {cameraMode === "pantry"
                  ? "Snap a photo of your open fridge"
                  : "Place barcode inside the box"}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
