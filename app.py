import streamlit as st
import cv2
import numpy as np
import time
import io
from PIL import Image
from fusion_logic import run_fusion

st.set_page_config(
    page_title="LG Fusion — Neural Image Fusion",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── TOKENS ── */
:root {
  --bg:      #F8F7F4;
  --s1:      #F1F0EB;
  --s2:      #E9E7E0;
  --s3:      #D2CFCA;
  --s4:      #9C9990;
  --s5:      #5E5C57;
  --s6:      #282624;
  --ink:     #0E0D0C;
  --white:   #FFFFFF;
  --gold:    #BFA46A;
  --gold2:   #9C8448;
  --gold-bg: rgba(191,164,106,.10);
  --gold-bd: rgba(191,164,106,.26);
  --red:     #D95F4B;
  --blue:    #4A72B8;
  --green:   #4A9A68;
  --r:    12px;
  --r-lg: 20px;
  --sh1: 0 1px 3px rgba(14,13,12,.05),0 2px 10px rgba(14,13,12,.06);
  --sh2: 0 4px 20px rgba(14,13,12,.09),0 1px 4px rgba(14,13,12,.05);
  --sh3: 0 16px 56px rgba(14,13,12,.13);
}

/* ── NUCLEAR LIGHT OVERRIDE ── */
html,body,.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],.main,
[data-testid="stAppViewContainer"]>section{
  background:var(--bg)!important; color:var(--s6)!important;
}
[data-testid="stVerticalBlock"],[data-testid="stHorizontalBlock"],
[data-testid="stColumn"],[data-testid="element-container"],
[data-testid="stMarkdownContainer"],[class*="st-emotion-cache"],
.block-container,section.main>div{
  background:transparent!important; color:var(--s6)!important;
}
[data-testid="stHeader"],[data-testid="stToolbar"],
[data-testid="collapsedControl"],header[data-testid="stHeader"]{display:none!important;}
[data-testid="stSidebar"]{background:var(--s1)!important;}
.block-container{padding:0!important;max-width:100%!important;}
[data-testid="stAppViewContainer"]>.main{padding:0!important;}
p,h1,h2,h3,h4,h5,h6,span,div,label,li,td,th,small,strong,em{color:var(--s6)!important;}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"]{background:transparent!important;}
[data-testid="stFileUploaderDropzone"]{
  background:var(--s1)!important;
  border:1.5px dashed var(--s3)!important;
  border-radius:var(--r)!important;
  transition:all .2s!important;
}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploaderDropzone"]:focus-within{
  border-color:var(--gold)!important;
  background:var(--gold-bg)!important;
  box-shadow:0 0 0 3px var(--gold-bd)!important;
}
[data-testid="stFileUploaderDropzone"] svg{fill:var(--s3)!important;stroke:var(--s3)!important;}
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] *{
  color:var(--s4)!important;font-family:'Outfit',sans-serif!important;font-size:13px!important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label *{
  color:var(--s5)!important;font-family:'Outfit',sans-serif!important;
  font-size:11px!important;font-weight:500!important;letter-spacing:.06em!important;
}
[data-testid="stFileUploaderFile"],[data-testid="stFileUploaderFileName"]{
  background:var(--white)!important;border:1px solid var(--s2)!important;
  border-radius:8px!important;color:var(--s6)!important;
}
[data-testid="stFileUploaderFileData"],
[data-testid="stFileUploaderFileData"] *{color:var(--s5)!important;}

/* ══════════════════════════════════════════════
   PRIMARY BUTTON — BLACK BG, WHITE TEXT, ALWAYS
══════════════════════════════════════════════ */
.stButton>button{
  background: #0E0D0C !important;
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
  border: 2px solid #0E0D0C !important;
  border-radius: var(--r) !important;
  padding: 18px 40px !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 15px !important;
  font-weight: 600 !important;
  letter-spacing: .08em !important;
  text-transform: uppercase !important;
  width: 100% !important;
  transition: all .22s ease !important;
  box-shadow: 0 4px 24px rgba(14,13,12,.28) !important;
  cursor: pointer !important;
  opacity: 1 !important;
}
.stButton>button *,
.stButton>button p,
.stButton>button span,
.stButton>button div{
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
}
.stButton>button:hover:not(:disabled){
  background: #282624 !important;
  border-color: #282624 !important;
  box-shadow: 0 8px 36px rgba(14,13,12,.36) !important;
  transform: translateY(-2px) !important;
}
.stButton>button:active:not(:disabled){transform:translateY(0)!important;}
.stButton>button:disabled{
  background: var(--s2) !important;
  color: var(--s4) !important;
  -webkit-text-fill-color: var(--s4) !important;
  border-color: var(--s2) !important;
  box-shadow: none !important;
  cursor: not-allowed !important;
}
.stButton>button:disabled *,
.stButton>button:disabled span{
  color: var(--s4) !important;
  -webkit-text-fill-color: var(--s4) !important;
}

/* ── DOWNLOAD BUTTONS ── */
.stDownloadButton>button{
  background:var(--white)!important;color:var(--ink)!important;
  -webkit-text-fill-color:var(--ink)!important;
  border:1.5px solid var(--s2)!important;border-radius:var(--r)!important;
  padding:13px 24px!important;font-family:'Outfit',sans-serif!important;
  font-size:13px!important;font-weight:500!important;
  width:100%!important;transition:all .2s!important;box-shadow:var(--sh1)!important;
}
.stDownloadButton>button *,.stDownloadButton>button span{
  color:var(--ink)!important;-webkit-text-fill-color:var(--ink)!important;
}
.stDownloadButton>button:hover{
  border-color:var(--gold)!important;background:var(--gold-bg)!important;
  box-shadow:0 4px 20px rgba(191,164,106,.20)!important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"]{
  background:var(--s1)!important;border:1px solid var(--s2)!important;
  border-radius:var(--r)!important;box-shadow:none!important;overflow:hidden!important;
}
[data-testid="stExpander"]>details,[data-testid="stExpander"] details[open]{background:var(--s1)!important;}
[data-testid="stExpander"] summary{
  background:var(--s1)!important;color:var(--s6)!important;
  font-family:'Outfit',sans-serif!important;font-size:13px!important;
  font-weight:500!important;padding:16px 20px!important;
}
[data-testid="stExpander"] summary:hover{background:var(--bg)!important;}
[data-testid="stExpander"] summary svg{stroke:var(--s4)!important;fill:var(--s4)!important;}
[data-testid="stExpanderDetails"]{
  background:var(--s1)!important;padding:20px!important;border-top:1px solid var(--s2)!important;
}
[data-testid="stExpanderDetails"] *{color:var(--s5)!important;}
[data-testid="stExpanderDetails"] strong{color:var(--s6)!important;font-weight:600!important;}
[data-testid="stExpanderDetails"] code{
  background:var(--s2)!important;color:var(--s6)!important;border-radius:4px!important;
  padding:2px 6px!important;font-family:'JetBrains Mono',monospace!important;font-size:11px!important;
}

/* ── MISC ── */
[data-testid="stImage"]{background:transparent!important;}
[data-testid="stImage"] img{display:block!important;}
[data-testid="stSpinner"]>div,.stSpinner>div{
  border-color:var(--s2) var(--s2) var(--s2) var(--ink)!important;
}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--s3);border-radius:4px;}

/* ════════════════════════════════════════════════
   CUSTOM LAYOUT COMPONENTS
════════════════════════════════════════════════ */

/* NAV */
.Q-nav{
  position:sticky;top:0;z-index:9000;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 60px;height:60px;
  background:rgba(248,247,244,.96);
  backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);
  border-bottom:1px solid var(--s2);
}
.Q-logo{
  font-family:'Playfair Display',serif;font-size:20px;font-weight:700;
  letter-spacing:.02em;color:var(--ink)!important;display:flex;align-items:center;gap:6px;
}
.Q-logo-sep{color:var(--gold);font-style:italic;font-weight:400;}
.Q-nav-pills{display:flex;gap:2px;}
.Q-pill{
  font-family:'Outfit',sans-serif;font-size:11px;font-weight:500;
  letter-spacing:.1em;text-transform:uppercase;
  color:var(--s4)!important;padding:6px 14px;border-radius:100px;
  border:1px solid transparent;
}
.Q-pill-on{color:var(--s6)!important;background:var(--s1);border-color:var(--s2);}
.Q-ver{
  font-family:'JetBrains Mono',monospace;font-size:10px;
  color:var(--s4)!important;background:var(--s1);
  border:1px solid var(--s2);padding:4px 12px;border-radius:100px;
}

/* HERO */
.Q-hero{
  position:relative;overflow:hidden;
  padding:72px 60px 60px;
  background:linear-gradient(145deg,#F8F7F4 0%,#F2F1EC 45%,#EAE8E0 100%);
  border-bottom:1px solid var(--s2);
}
.Q-glow-a{
  position:absolute;top:-100px;right:-60px;
  width:500px;height:500px;border-radius:50%;
  background:radial-gradient(circle,rgba(191,164,106,.13) 0%,transparent 65%);
  pointer-events:none;
}
.Q-glow-b{
  position:absolute;bottom:-60px;left:20%;
  width:280px;height:280px;border-radius:50%;
  background:radial-gradient(circle,rgba(191,164,106,.08) 0%,transparent 70%);
  pointer-events:none;
}
.Q-chip{
  display:inline-flex;align-items:center;gap:7px;
  font-family:'JetBrains Mono',monospace;font-size:10px;
  letter-spacing:.15em;text-transform:uppercase;
  color:var(--gold)!important;
  background:rgba(191,164,106,.09);border:1px solid rgba(191,164,106,.24);
  padding:5px 14px;border-radius:100px;margin-bottom:22px;
}
.Q-h1{
  font-family:'Playfair Display',serif;
  font-size:clamp(46px,6vw,84px);font-weight:700;
  line-height:1.03;letter-spacing:-.02em;
  color:var(--ink)!important;margin:0 0 16px;
}
.Q-h1 i{font-weight:400;color:var(--gold);}
.Q-hero-sub{
  font-family:'Outfit',sans-serif;font-size:15px;font-weight:300;line-height:1.75;
  color:var(--s4)!important;max-width:560px;margin:0;
}
.Q-stats{display:flex;gap:44px;margin-top:44px;flex-wrap:wrap;}
.Q-stat{display:flex;flex-direction:column;gap:3px;}
.Q-sv{
  font-family:'Playfair Display',serif;font-size:33px;font-weight:700;
  color:var(--ink)!important;line-height:1;
}
.Q-sl{
  font-family:'Outfit',sans-serif;font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--s4)!important;
}

/* PIPELINE */
.Q-pipe-wrap{
  background:var(--s1);border-bottom:1px solid var(--s2);
  padding:20px 60px;
}
.Q-pipe{display:flex;align-items:center;overflow-x:auto;padding-bottom:2px;}
.Q-pn{display:flex;flex-direction:column;align-items:center;gap:7px;flex:1;min-width:72px;}
.Q-pi{
  width:44px;height:44px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;font-size:17px;
}
.pi-r{background:rgba(217,95,75,.10);border:1px solid rgba(217,95,75,.25);}
.pi-b{background:rgba(74,114,184,.10);border:1px solid rgba(74,114,184,.25);}
.pi-p{background:rgba(120,90,190,.09);border:1px solid rgba(120,90,190,.22);}
.pi-g{background:rgba(191,164,106,.14);border:1px solid rgba(191,164,106,.30);}
.pi-e{background:rgba(74,154,104,.10);border:1px solid rgba(74,154,104,.24);}
.Q-pl{
  font-family:'Outfit',sans-serif;font-size:9px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--s4)!important;text-align:center;line-height:1.4;
}
.Q-pa{color:var(--s3)!important;font-size:20px;padding-bottom:20px;flex-shrink:0;margin:0 4px;}

/* ════════ NEW LAYOUT: TOP→BOTTOM FLOW ════════ */

/* STEP HEADER */
.Q-step{
  display:flex;align-items:center;gap:14px;margin-bottom:20px;
}
.Q-step-num{
  width:36px;height:36px;border-radius:50%;
  background:var(--ink);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
}
.Q-step-num span{
  font-family:'Outfit',sans-serif;font-size:13px;font-weight:700;
  color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important;line-height:1;
}
.Q-step-info{}
.Q-step-title{
  font-family:'Playfair Display',serif;font-size:22px;font-weight:700;
  color:var(--ink)!important;margin:0 0 2px;line-height:1.1;
}
.Q-step-desc{
  font-family:'Outfit',sans-serif;font-size:12px;font-weight:300;
  color:var(--s4)!important;margin:0;letter-spacing:.02em;
}

/* UPLOAD GRID — 2 side-by-side uploaders */
.Q-upload-grid{
  display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;
}
.Q-upload-slot{
  background:var(--white);border:1px solid var(--s2);
  border-radius:var(--r-lg);overflow:hidden;box-shadow:var(--sh1);
}
.Q-upload-slot-head{
  padding:14px 18px;background:var(--s1);
  border-bottom:1px solid var(--s2);
  display:flex;align-items:center;gap:10px;
}
.Q-upload-slot-label{
  font-family:'Outfit',sans-serif;font-size:11px;font-weight:600;
  letter-spacing:.1em;text-transform:uppercase;color:var(--s5)!important;
}
.Q-dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0;}
.Q-upload-slot-body{padding:12px;}

/* PREVIEW STRIP — 2 images side by side with metadata */
.Q-prev-strip{
  display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px;
}
.Q-prev-card{
  background:var(--white);border:1px solid var(--s2);
  border-radius:var(--r-lg);overflow:hidden;box-shadow:var(--sh1);
  transition:box-shadow .22s;
}
.Q-prev-card:hover{box-shadow:var(--sh2);}
.Q-prev-head{
  padding:12px 16px;background:var(--s1);border-bottom:1px solid var(--s2);
  display:flex;align-items:center;justify-content:space-between;
}
.Q-prev-title{
  font-family:'Outfit',sans-serif;font-size:10px;font-weight:600;
  letter-spacing:.1em;text-transform:uppercase;color:var(--s5)!important;
  display:flex;align-items:center;gap:7px;
}
.Q-prev-meta{
  font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--s3)!important;
  display:flex;gap:6px;
}
.Q-prev-badge{
  background:var(--s2);border-radius:4px;padding:2px 6px;
  font-size:9px;color:var(--s4)!important;
}
.Q-prev-body{padding:10px;}
.Q-prev-body img{border-radius:8px;}

/* RUN BUTTON SECTION */
.Q-run-section{
  background:var(--white);
  border:1px solid var(--s2);
  border-radius:var(--r-lg);
  padding:28px 32px;
  box-shadow:var(--sh1);
  margin-bottom:40px;
  display:flex;align-items:center;gap:24px;
}
.Q-run-info{flex:1;}
.Q-run-title{
  font-family:'Playfair Display',serif;font-size:18px;font-weight:700;
  color:var(--ink)!important;margin:0 0 4px;
}
.Q-run-desc{
  font-family:'Outfit',sans-serif;font-size:12px;font-weight:300;
  color:var(--s4)!important;margin:0;
}
.Q-run-chips{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap;}
.Q-rchip{
  font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.08em;
  color:var(--s4)!important;background:var(--s1);
  border:1px solid var(--s2);padding:3px 9px;border-radius:100px;
}
.Q-run-btn-wrap{flex-shrink:0;width:220px;}

/* SECTION DIVIDER */
.Q-divider{
  display:flex;align-items:center;gap:14px;margin:32px 0;
}
.Q-divider-line{flex:1;height:1px;background:var(--s2);}
.Q-divider-label{
  font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--gold)!important;
  background:var(--gold-bg);border:1px solid var(--gold-bd);
  padding:4px 12px;border-radius:100px;white-space:nowrap;
}

/* METRICS GRID — 4 across */
.Q-metrics{
  display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;
}
.Q-m{
  background:var(--white);border:1px solid var(--s2);
  border-radius:var(--r-lg);padding:22px 16px 18px;text-align:center;
  box-shadow:var(--sh1);transition:all .22s;
  position:relative;overflow:hidden;
}
.Q-m::after{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--gold),var(--gold2));
  opacity:0;transition:opacity .2s;
}
.Q-m:hover{box-shadow:var(--sh2);transform:translateY(-2px);}
.Q-m:hover::after{opacity:1;}
.Q-mv{
  font-family:'Playfair Display',serif;font-size:32px;font-weight:700;
  color:var(--ink)!important;line-height:1;
}
.Q-ml{
  font-family:'Outfit',sans-serif;font-size:9px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--s4)!important;margin-top:7px;
}
.Q-ms{
  font-family:'JetBrains Mono',monospace;font-size:9px;
  color:var(--s3)!important;margin-top:4px;
}

/* OUTPUT CARD — full width fused result */
.Q-outcard{
  background:var(--white);border:1px solid var(--s2);
  border-radius:var(--r-lg);overflow:hidden;box-shadow:var(--sh2);
  margin-bottom:20px;
  animation:Q-up .4s ease both;
}
.Q-outcard-head{
  padding:20px 26px;
  background:linear-gradient(90deg,var(--s1) 0%,var(--bg) 100%);
  border-bottom:1px solid var(--s2);
  display:flex;align-items:center;justify-content:space-between;
}
.Q-outcard-title{
  font-family:'Playfair Display',serif;font-size:22px;font-weight:700;
  color:var(--ink)!important;display:flex;align-items:center;gap:10px;
}
.Q-outcard-badges{display:flex;gap:6px;}
.Q-badge{
  font-family:'JetBrains Mono',monospace;font-size:9px;
  letter-spacing:.1em;text-transform:uppercase;
  padding:4px 10px;border-radius:100px;
}
.Q-bg{color:var(--gold2)!important;background:var(--gold-bg);border:1px solid var(--gold-bd);}
.Q-bs{color:var(--s4)!important;background:var(--s1);border:1px solid var(--s2);}
.Q-outcard-body{padding:20px;}

/* COMPARE GRID — 3 equal thumbnails */
.Q-compare-grid{
  display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px;
  animation:Q-up .45s .06s ease both;
}
.Q-thumb{
  background:var(--white);border:1px solid var(--s2);
  border-radius:var(--r-lg);overflow:hidden;box-shadow:var(--sh1);
  transition:all .22s;
}
.Q-thumb:hover{box-shadow:var(--sh2);transform:translateY(-2px);}
.Q-thumb-head{
  padding:11px 16px;background:var(--s1);border-bottom:1px solid var(--s2);
  display:flex;align-items:center;gap:7px;
}
.Q-thumb-label{
  font-family:'Outfit',sans-serif;font-size:10px;font-weight:600;
  letter-spacing:.1em;text-transform:uppercase;color:var(--s5)!important;
}
.Q-thumb-body{padding:10px;}
.Q-thumb-body img{border-radius:8px;}

/* DOWNLOAD SECTION */
.Q-dl-section{
  display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px;
  animation:Q-up .5s .1s ease both;
}

/* ARCH SECTION */
.Q-arch{
  background:var(--s1);border:1px solid var(--s2);
  border-radius:var(--r-lg);overflow:hidden;
  animation:Q-up .55s .14s ease both;
}
.Q-arch-blocks{
  display:grid;grid-template-columns:repeat(3,1fr);gap:0;
}
.Q-arch-block{
  padding:22px 20px;border-right:1px solid var(--s2);
}
.Q-arch-block:last-child{border-right:none;}
.Q-arch-block-title{
  font-family:'Outfit',sans-serif;font-size:11px;font-weight:700;
  letter-spacing:.08em;text-transform:uppercase;color:var(--ink)!important;
  display:flex;align-items:center;gap:8px;margin-bottom:14px;
  padding-bottom:10px;border-bottom:1px solid var(--s2);
}
.Q-ai{
  font-family:'Outfit',sans-serif;font-size:12px;font-weight:300;
  color:var(--s5)!important;display:flex;gap:8px;
  align-items:flex-start;margin-bottom:7px;line-height:1.5;
}
.Q-ai::before{content:'·';color:var(--gold)!important;flex-shrink:0;margin-top:1px;}
.Q-acode{
  font-family:'JetBrains Mono',monospace;font-size:10px;
  background:var(--s2);color:var(--s6)!important;
  border-radius:4px;padding:1px 6px;
}

/* IDLE / PROCESSING */
.Q-idle{
  padding:72px 40px;text-align:center;
  background:var(--s1);border:1.5px dashed var(--s3);
  border-radius:var(--r-lg);margin-bottom:20px;
}
.Q-idle-icon{font-size:42px;opacity:.22;display:block;margin-bottom:16px;}
.Q-idle-title{
  font-family:'Playfair Display',serif;font-size:24px;font-weight:400;
  color:var(--s4)!important;margin:0 0 8px;
}
.Q-idle-sub{
  font-family:'JetBrains Mono',monospace;font-size:11px;letter-spacing:.07em;
  color:var(--s3)!important;
}
.Q-proc{
  padding:52px 32px;text-align:center;
  background:var(--s1);border:1px solid var(--s2);
  border-radius:var(--r-lg);margin-bottom:20px;
}
.Q-proc-title{
  font-family:'Playfair Display',serif;font-size:22px;font-weight:400;
  color:var(--s6)!important;margin-bottom:10px;
}
.Q-proc-sub{
  font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.08em;
  color:var(--s4)!important;
}
.Q-dots{display:inline-flex;gap:7px;margin-top:22px;}
.Q-dot-anim{
  width:7px;height:7px;border-radius:50%;
  background:var(--gold);animation:Q-pulse 1.4s ease-in-out infinite;
}
.Q-dot-anim:nth-child(2){animation-delay:.2s;}
.Q-dot-anim:nth-child(3){animation-delay:.4s;}
@keyframes Q-pulse{
  0%,80%,100%{transform:scale(.7);opacity:.4;}
  40%{transform:scale(1.2);opacity:1;}
}

/* HINT */
.Q-hint{
  text-align:center;margin-top:10px;
  font-family:'JetBrains Mono',monospace;font-size:11px;
  color:var(--s4)!important;letter-spacing:.06em;
}

/* FOOTER */
.Q-footer{
  margin-top:48px;padding:26px 60px;
  border-top:1px solid var(--s2);
  display:flex;align-items:center;justify-content:space-between;
  flex-wrap:wrap;gap:12px;background:var(--bg)!important;
}
.Q-footer-logo{
  font-family:'Playfair Display',serif;font-size:16px;font-weight:700;
  color:var(--s6)!important;
}
.Q-footer-logo em{font-style:italic;font-weight:400;color:var(--gold);}
.Q-ftags{display:flex;gap:6px;flex-wrap:wrap;}
.Q-ftag{
  font-family:'JetBrains Mono',monospace;font-size:10px;
  color:var(--s4)!important;background:var(--s1);
  border:1px solid var(--s2);padding:3px 10px;border-radius:100px;
}

/* CONTAINER */
.Q-wrap{max-width:1180px;margin:0 auto;padding:40px 60px 0;}
@media(max-width:960px){
  .Q-wrap{padding:24px 24px 0;}
  .Q-upload-grid,.Q-prev-strip,.Q-compare-grid,.Q-arch-blocks{grid-template-columns:1fr 1fr;}
  .Q-metrics{grid-template-columns:repeat(2,1fr);}
  .Q-run-section{flex-direction:column;align-items:stretch;}
  .Q-run-btn-wrap{width:100%;}
}

/* TEAM PANEL — right side of hero */
.Q-hero-inner{
  display:grid;grid-template-columns:1fr auto;gap:40px;align-items:start;
}
.Q-team-panel{
  background:rgba(255,255,255,.62);
  border:1px solid var(--s2);
  border-radius:var(--r-lg);
  padding:22px 24px;
  min-width:220px;
  backdrop-filter:blur(10px);
  box-shadow:var(--sh1);
  flex-shrink:0;
}
.Q-team-eyebrow{
  font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--gold)!important;
  margin-bottom:14px;display:flex;align-items:center;gap:8px;
}
.Q-team-eyebrow::after{
  content:'';flex:1;height:1px;background:var(--s2);
}
.Q-team-member{
  display:flex;align-items:center;gap:10px;
  padding:8px 0;border-bottom:1px solid var(--s2);
}
.Q-team-member:last-child{border-bottom:none;padding-bottom:0;}
.Q-team-member:first-of-type{padding-top:0;}
.Q-team-avatar{
  width:30px;height:30px;border-radius:50%;
  background:linear-gradient(135deg,var(--gold-bg),var(--s2));
  border:1px solid var(--gold-bd);
  display:flex;align-items:center;justify-content:center;
  font-family:'Playfair Display',serif;font-size:12px;font-weight:700;
  color:var(--gold2)!important;flex-shrink:0;
}
.Q-team-name{
  font-family:'Outfit',sans-serif;font-size:12px;font-weight:500;
  color:var(--s6)!important;line-height:1.3;
}
.Q-team-name span{
  display:block;font-size:10px;font-weight:300;
  color:var(--s4)!important;letter-spacing:.03em;
}

/* IMAGE SIZE CONTROL — cap preview and output images */
.Q-prev-body img,
.Q-prev-body [data-testid="stImage"] img{
  max-height:220px !important;
  object-fit:cover !important;
  width:100% !important;
}
.Q-outcard-body img,
.Q-outcard-body [data-testid="stImage"] img{
  max-height:360px !important;
  object-fit:contain !important;
  width:100% !important;
}
.Q-thumb-body img,
.Q-thumb-body [data-testid="stImage"] img{
  max-height:160px !important;
  object-fit:cover !important;
  width:100% !important;
}
/* Streamlit image wrappers */
.Q-prev-body [data-testid="stImage"],
.Q-outcard-body [data-testid="stImage"],
.Q-thumb-body [data-testid="stImage"]{
  overflow:hidden !important;
  border-radius:8px !important;
}

/* ANIMATIONS */
@keyframes Q-up{
  from{opacity:0;transform:translateY(16px);}
  to{opacity:1;transform:translateY(0);}
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def read_uploaded(f) -> np.ndarray:
    raw = np.asarray(bytearray(f.read()), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)

def to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr)
    return Image.fromarray(arr)

def arr_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype == np.uint8 and len(arr.shape) == 3 and arr.shape[2] == 3:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    return to_pil(arr)

def to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def compute_metrics(fused: np.ndarray, vis: np.ndarray):
    f32   = fused.astype(np.float32) / 255.0
    vis_r = cv2.resize(vis, (fused.shape[1], fused.shape[0]))
    psnr  = cv2.PSNR(fused, vis_r)
    ent   = float(-np.sum(np.where(f32 > 0, f32 * np.log2(f32 + 1e-9), 0)) / f32.size * 10)
    std   = float(np.std(f32))
    return round(float(psnr), 1), round(ent, 2), round(std, 3)


# ═══════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════
for k, v in [("result", None), ("meta", {}), ("ir_arr", None), ("vis_arr", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
#  NAV
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="Q-nav">
  <div class="Q-logo">LG<span class="Q-logo-sep">&nbsp;·&nbsp;</span>Fusion</div>
  <div class="Q-nav-pills">
    <div class="Q-pill Q-pill-on">Workspace</div>
    <div class="Q-pill">Architecture</div>
    <div class="Q-pill">Docs</div>
  </div>
  <div class="Q-ver">v2.0 &nbsp;·&nbsp; Restormer + CIIM</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="Q-hero">
  <div class="Q-glow-a"></div>
  <div class="Q-glow-b"></div>
  <div class="Q-hero-inner">
    <div>
      <div class="Q-chip">✦ &nbsp; Neural Image Fusion System</div>
      <h1 class="Q-h1">Infrared &times; Visible<br><i>Fusion Engine</i></h1>
      <p class="Q-hero-sub">
        Restormer encoder–decoder with wavelet convolutions, frequency-domain
        attention modules, and the CIIM cross-modal interaction block —
        delivering perceptually superior fused imagery.
      </p>
      <div class="Q-stats">
        <div class="Q-stat"><span class="Q-sv">256²</span><span class="Q-sl">Resolution</span></div>
        <div class="Q-stat"><span class="Q-sv">6ch</span><span class="Q-sl">Input Channels</span></div>
        <div class="Q-stat"><span class="Q-sv">CIIM</span><span class="Q-sl">Fusion Block</span></div>
        <div class="Q-stat"><span class="Q-sv">YCrCb</span><span class="Q-sl">Color Space</span></div>
      </div>
    </div>
    <div class="Q-team-panel">
      <div class="Q-team-eyebrow">Research Team</div>
      <div class="Q-team-member">
        <div class="Q-team-avatar">A</div>
        <div class="Q-team-name">Afrin Nisha. B<span>Team Leader</span></div>
      </div>
      <div class="Q-team-member">
        <div class="Q-team-avatar">G</div>
        <div class="Q-team-name">Gnana Saraswathi. G<span>Team Member</span></div>
      </div>
      <div class="Q-team-member">
        <div class="Q-team-avatar">A</div>
        <div class="Q-team-name">Anushiya Devi. M<span>Team Member</span></div>
      </div>
      <div class="Q-team-member">
        <div class="Q-team-avatar">K</div>
        <div class="Q-team-name">Kaviya. S<span>Team Member</span></div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="Q-pipe-wrap">
  <div class="Q-pipe">
    <div class="Q-pn"><div class="Q-pi pi-r">🌡️</div><div class="Q-pl">Infrared<br>Input</div></div>
    <div class="Q-pa">→</div>
    <div class="Q-pn"><div class="Q-pi pi-p">⚡</div><div class="Q-pl">Patch<br>Embed</div></div>
    <div class="Q-pa">→</div>
    <div class="Q-pn"><div class="Q-pi pi-p">🔬</div><div class="Q-pl">Base+Detail<br>Extract</div></div>
    <div class="Q-pa">→</div>
    <div class="Q-pn"><div class="Q-pi pi-g">✦</div><div class="Q-pl">CIIM<br>Fusion</div></div>
    <div class="Q-pa">→</div>
    <div class="Q-pn"><div class="Q-pi pi-e">🖼️</div><div class="Q-pl">Fused<br>Output</div></div>
    <div class="Q-pa">←</div>
    <div class="Q-pn"><div class="Q-pi pi-b">📷</div><div class="Q-pl">Visible<br>Input</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN WRAPPER
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="Q-wrap">', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  STEP 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="Q-step">
  <div class="Q-step-num"><span>01</span></div>
  <div class="Q-step-info">
    <div class="Q-step-title">Upload Source Images</div>
    <div class="Q-step-desc">Co-registered infrared and visible-light pair &nbsp;·&nbsp; PNG · JPG · JPEG · BMP</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Two-column upload zone
up_l, up_r = st.columns(2, gap="medium")

with up_l:
    st.markdown("""
    <div class="Q-upload-slot">
      <div class="Q-upload-slot-head">
        <span class="Q-dot" style="background:#D95F4B"></span>
        <span class="Q-upload-slot-label">Infrared Image</span>
      </div>
      <div class="Q-upload-slot-body">
    """, unsafe_allow_html=True)
    ir_file = st.file_uploader(
        "IR", type=["png","jpg","jpeg","bmp"],
        key="ir_up", label_visibility="collapsed",
    )
    st.markdown("</div></div>", unsafe_allow_html=True)

with up_r:
    st.markdown("""
    <div class="Q-upload-slot">
      <div class="Q-upload-slot-head">
        <span class="Q-dot" style="background:#4A72B8"></span>
        <span class="Q-upload-slot-label">Visible-Light Image</span>
      </div>
      <div class="Q-upload-slot-body">
    """, unsafe_allow_html=True)
    vis_file = st.file_uploader(
        "VIS", type=["png","jpg","jpeg","bmp"],
        key="vis_up", label_visibility="collapsed",
    )
    st.markdown("</div></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PREVIEW STRIP — shown when at least one uploaded
# ═══════════════════════════════════════════════════════════════
if ir_file or vis_file:
    st.markdown("<br>", unsafe_allow_html=True)
    pv_l, pv_r = st.columns(2, gap="medium")

    if ir_file:
        ir_file.seek(0)
        ir_pil = Image.open(ir_file)
        w, h = ir_pil.size
        ir_display = ir_pil.copy()
        ir_display.thumbnail((480, 300), Image.LANCZOS)
        with pv_l:
            st.markdown(f"""
            <div class="Q-prev-card">
              <div class="Q-prev-head">
                <span class="Q-prev-title">
                  <span class="Q-dot" style="background:#D95F4B"></span>
                  Infrared
                </span>
                <span class="Q-prev-meta">
                  <span class="Q-prev-badge">{w}×{h}</span>
                  <span class="Q-prev-badge">IR</span>
                </span>
              </div>
              <div class="Q-prev-body">
            """, unsafe_allow_html=True)
            st.image(ir_display, use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    if vis_file:
        vis_file.seek(0)
        vis_pil = Image.open(vis_file)
        w2, h2 = vis_pil.size
        vis_display = vis_pil.copy()
        vis_display.thumbnail((480, 300), Image.LANCZOS)
        with pv_r:
            st.markdown(f"""
            <div class="Q-prev-card">
              <div class="Q-prev-head">
                <span class="Q-prev-title">
                  <span class="Q-dot" style="background:#4A72B8"></span>
                  Visible
                </span>
                <span class="Q-prev-meta">
                  <span class="Q-prev-badge">{w2}×{h2}</span>
                  <span class="Q-prev-badge">RGB</span>
                </span>
              </div>
              <div class="Q-prev-body">
            """, unsafe_allow_html=True)
            st.image(vis_display, use_container_width=True)
            st.markdown("</div></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  STEP 2 — RUN FUSION (horizontal card layout)
# ═══════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="Q-step">
  <div class="Q-step-num"><span>02</span></div>
  <div class="Q-step-info">
    <div class="Q-step-title">Run Neural Fusion</div>
    <div class="Q-step-desc">Restormer encoder · CIIM block · YCrCb color fusion</div>
  </div>
</div>
""", unsafe_allow_html=True)

ready = bool(ir_file and vis_file)

# Run card — info on left, button on right
run_card_l, run_card_r = st.columns([2, 1], gap="medium")

with run_card_l:
    st.markdown("""
    <div style="background:var(--white);border:1px solid var(--s2);
                border-radius:var(--r-lg);padding:24px 28px;box-shadow:var(--sh1);">
      <div style="font-family:'Playfair Display',serif;font-size:18px;font-weight:700;
                  color:var(--ink);margin-bottom:6px;">Ready to Process</div>
      <div style="font-family:'Outfit',sans-serif;font-size:13px;font-weight:300;
                  color:var(--s4);margin-bottom:14px;line-height:1.6;">
        The fusion pipeline will encode both modalities, apply cross-modal
        interaction, decode into a unified image, and render quality metrics.
      </div>
      <div style="display:flex;gap:6px;flex-wrap:wrap;">
        <span class="Q-ftag">Restormer</span>
        <span class="Q-ftag">WTConv</span>
        <span class="Q-ftag">CIIM</span>
        <span class="Q-ftag">FFT Attention</span>
        <span class="Q-ftag">YCrCb α=0.6</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with run_card_r:
    run_btn = st.button(
        "✦  Run Neural Fusion" if ready else "⊘  Upload Images First",
        disabled=not ready,
        use_container_width=True,
    )
    if not ready:
        st.markdown(
            '<div class="Q-hint">↑ Upload both images above</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
#  STEP 3 — RESULTS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="Q-divider">
  <div class="Q-divider-line"></div>
  <div class="Q-divider-label">03 · Fusion Output</div>
  <div class="Q-divider-line"></div>
</div>
""", unsafe_allow_html=True)

result_slot = st.empty()

# ── Idle ──
if not run_btn and st.session_state.result is None:
    result_slot.markdown("""
    <div class="Q-idle">
      <span class="Q-idle-icon">✦</span>
      <div class="Q-idle-title">Awaiting Fusion</div>
      <div class="Q-idle-sub">Complete steps 01 &amp; 02 above to generate output</div>
    </div>
    """, unsafe_allow_html=True)

# ── Processing ──
if run_btn and ir_file and vis_file:
    result_slot.markdown("""
    <div class="Q-proc">
      <div class="Q-proc-title">Processing Neural Fusion…</div>
      <div class="Q-proc-sub">
        Restormer Encoder &nbsp;·&nbsp; CIIM Fusion Block &nbsp;·&nbsp; YCrCb Decoder
      </div>
      <div class="Q-dots">
        <div class="Q-dot-anim"></div>
        <div class="Q-dot-anim"></div>
        <div class="Q-dot-anim"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ir_file.seek(0);  vis_file.seek(0)
    ir_arr  = read_uploaded(ir_file)
    vis_arr = read_uploaded(vis_file)

    t0        = time.time()
    fused_arr = run_fusion(ir_arr, vis_arr)
    elapsed   = time.time() - t0

    psnr, ent, std = compute_metrics(fused_arr, vis_arr)

    st.session_state.result  = fused_arr
    st.session_state.meta    = {"psnr": psnr, "entropy": ent, "std": std, "time": round(elapsed, 2)}
    st.session_state.ir_arr  = ir_arr
    st.session_state.vis_arr = vis_arr
    result_slot.empty()
    st.rerun()

# ── Show results ──
if st.session_state.result is not None:
    fused     = st.session_state.result
    meta      = st.session_state.meta
    ir_arr    = st.session_state.ir_arr
    vis_arr   = st.session_state.vis_arr
    fused_pil = to_pil(fused)

    # ─ 4 metrics full-width ─
    st.markdown(f"""
    <div class="Q-metrics">
      <div class="Q-m">
        <div class="Q-mv">{meta['psnr']}</div>
        <div class="Q-ml">PSNR (dB)</div>
        <div class="Q-ms">vs visible</div>
      </div>
      <div class="Q-m">
        <div class="Q-mv">{meta['time']}s</div>
        <div class="Q-ml">Latency</div>
        <div class="Q-ms">inference time</div>
      </div>
      <div class="Q-m">
        <div class="Q-mv">{meta['entropy']}</div>
        <div class="Q-ml">Entropy</div>
        <div class="Q-ms">info density</div>
      </div>
      <div class="Q-m">
        <div class="Q-mv">{meta['std']}</div>
        <div class="Q-ml">Contrast σ</div>
        <div class="Q-ms">std deviation</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─ Full-width fused result card ─
    st.markdown("""
    <div class="Q-outcard">
      <div class="Q-outcard-head">
        <div class="Q-outcard-title">
          <span class="Q-dot" style="background:#BFA46A;width:10px;height:10px;"></span>
          Fused Output
        </div>
        <div class="Q-outcard-badges">
          <span class="Q-badge Q-bg">YCrCb</span>
          <span class="Q-badge Q-bs">α = 0.6</span>
          <span class="Q-badge Q-bs">256 × 256</span>
          <span class="Q-badge Q-bs">Restormer</span>
        </div>
      </div>
      <div class="Q-outcard-body">
    """, unsafe_allow_html=True)
    st.image(fused_pil, width=520)
    st.markdown("</div></div>", unsafe_allow_html=True)

    # ─ Side-by-side 3-way comparison ─
    st.markdown("""
    <div style="margin:20px 0 10px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.14em;
                  text-transform:uppercase;color:var(--gold);margin-bottom:12px;
                  display:flex;align-items:center;gap:10px;">
        Side-by-Side Comparison
        <span style="flex:1;height:1px;background:var(--s2);display:block;max-width:300px;"></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cmp_a, cmp_b, cmp_c = st.columns(3, gap="medium")

    with cmp_a:
        st.markdown("""
        <div class="Q-thumb">
          <div class="Q-thumb-head">
            <span class="Q-dot" style="background:#D95F4B"></span>
            <span class="Q-thumb-label">Infrared Input</span>
          </div>
          <div class="Q-thumb-body">
        """, unsafe_allow_html=True)
        st.image(arr_to_pil(ir_arr), use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with cmp_b:
        st.markdown("""
        <div class="Q-thumb">
          <div class="Q-thumb-head">
            <span class="Q-dot" style="background:#4A72B8"></span>
            <span class="Q-thumb-label">Visible Input</span>
          </div>
          <div class="Q-thumb-body">
        """, unsafe_allow_html=True)
        st.image(arr_to_pil(vis_arr), use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with cmp_c:
        st.markdown("""
        <div class="Q-thumb">
          <div class="Q-thumb-head">
            <span class="Q-dot" style="background:#BFA46A"></span>
            <span class="Q-thumb-label">Fused Output</span>
          </div>
          <div class="Q-thumb-body">
        """, unsafe_allow_html=True)
        st.image(fused_pil, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # ─ Download row ─
    st.markdown("<br>", unsafe_allow_html=True)
    dl_a, dl_b, dl_c = st.columns([1, 1, 1], gap="medium")
    with dl_a:
        st.download_button(
            "⬇  Export PNG",
            data=to_bytes(fused_pil, "PNG"),
            file_name="lgfusion_output.png",
            mime="image/png",
            use_container_width=True,
        )
    with dl_b:
        st.download_button(
            "⬇  Export JPEG",
            data=to_bytes(fused_pil, "JPEG"),
            file_name="lgfusion_output.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )
    with dl_c:
        st.download_button(
            "⬇  Export TIFF",
            data=to_bytes(fused_pil, "TIFF"),
            file_name="lgfusion_output.tiff",
            mime="image/tiff",
            use_container_width=True,
        )

    # ─ Architecture panel ─
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("✦  Architecture Details — Encoder · Fusion · Decoder"):
        st.markdown("""
        <div class="Q-arch-blocks">
          <div class="Q-arch-block">
            <div class="Q-arch-block-title">⚡ Encoder</div>
            <div class="Q-ai">Overlapping 3×3 patch embed → dim=64</div>
            <div class="Q-ai">4× <span class="Q-acode">TransformerBlock</span> (MDTA + GDFN)</div>
            <div class="Q-ai"><span class="Q-acode">BaseFeatureExtraction</span> → AdditiveTokenMixer + MLP</div>
            <div class="Q-ai"><span class="Q-acode">DetailFeatureExtraction</span> → MPE · FDCM · INN · DDIM · Sobel</div>
          </div>
          <div class="Q-arch-block">
            <div class="Q-arch-block-title">✦ CIIM Fusion Block</div>
            <div class="Q-ai">Dual Conv1 projection (IR + Vis branch)</div>
            <div class="Q-ai">DWConv + GELU / ReLU spatial gating</div>
            <div class="Q-ai">Channel shuffle (groups = dim // 2)</div>
            <div class="Q-ai">ECA attention (k = 1)</div>
          </div>
          <div class="Q-arch-block">
            <div class="Q-arch-block-title">🖼 Decoder</div>
            <div class="Q-ai">CIIM reduce → Conv1 → 4× TransformerBlock</div>
            <div class="Q-ai">Conv → LeakyReLU → Conv → Sigmoid + skip</div>
            <div class="Q-ai">Y = 0.4·Y_vis + 0.6·Y_ir (luminance blend)</div>
            <div class="Q-ai">Chrominance Cr, Cb preserved from visible</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# Close wrapper
st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="Q-footer">
  <div class="Q-footer-logo">LG<em> · </em>Fusion</div>
  <div class="Q-ftags">
    <span class="Q-ftag">Restormer</span>
    <span class="Q-ftag">WTConv</span>
    <span class="Q-ftag">CIIM</span>
    <span class="Q-ftag">FFT Attention</span>
    <span class="Q-ftag">YCrCb</span>
  </div>
  <div class="Q-ftags">
    <span class="Q-ftag">Streamlit</span>
    <span class="Q-ftag">TensorFlow</span>
    <span class="Q-ftag">OpenCV</span>
  </div>
</div>
""", unsafe_allow_html=True)