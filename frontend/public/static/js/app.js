/* ═══════════════════════════════════════════════════════════════════════
   Healthkaton — app.js
   All client-side logic: navigation, calculations, UI interactions.
   ═══════════════════════════════════════════════════════════════════════ */

'use strict';

// ── App State ────────────────────────────────────────────────────────────────
var STATE = {
  page:        'prediction',
  // Prediction inputs
  glucose:     100, bloodPressure: 100, age: 100,
  weight:      100, height: 100,
  gender:      'Male', activity:  'Sedentary (No exercise)',
  // Computed
  bmi: null, bmr: null, daily: null,
  bmr1: null, bmr2: null, bmr3: null,
  isDiabetic:  false, bmiCategory: '', bmiColor: '',
  // Advice text
  healthRows:  [], adviceLines: [], conclusion: '',
  dataString:  '',
  // Food recommendation
  recommended: { breakfast: [], lunch: [], dinner: [] },
  // Image upload
  uploadedImage: null,
  previewUrl: null,
};

const ACTIVITY_LABELS = [
  'Sedentary (No exercise)',
  'Light (exercise 1–2 times per week)',
  'Moderate (exercise 3–4 times per week)',
  'Active (exercise 3–5 times per week)',
  'Very Active (exercise 6–7 times per week)',
  'Intense',
];
const ACTIVITY_MULTIPLIERS = [1.0, 1.2, 1.375, 1.55, 1.725, 1.9];

// ── Sample food database ─────────────────────────────────────────────────────
const FOOD_DB = [
  // Breakfast options
  { name:'Oatmeal with Banana',    category:'breakfast', diabeticFriendly:true,  calories:320, fat:5,  satFat:1, chol:0,  sodium:50,  carbs:60, fiber:6, sugar:12, protein:9  },
  { name:'Greek Yogurt & Berries', category:'breakfast', diabeticFriendly:true,  calories:280, fat:4,  satFat:2, chol:15, sodium:80,  carbs:38, fiber:3, sugar:18, protein:18 },
  { name:'Whole Wheat Pancakes',   category:'breakfast', diabeticFriendly:true,  calories:350, fat:8,  satFat:2, chol:55, sodium:390, carbs:58, fiber:5, sugar:10, protein:12 },
  { name:'Scrambled Eggs & Toast', category:'breakfast', diabeticFriendly:true,  calories:390, fat:14, satFat:4, chol:370,sodium:460, carbs:38, fiber:3, sugar:4,  protein:20 },
  { name:'Avocado Toast',          category:'breakfast', diabeticFriendly:true,  calories:340, fat:18, satFat:3, chol:0,  sodium:290, carbs:40, fiber:8, sugar:3,  protein:9  },
  { name:'Pancakes with Syrup',    category:'breakfast', diabeticFriendly:false, calories:520, fat:12, satFat:4, chol:65, sodium:600, carbs:92, fiber:2, sugar:42, protein:10 },
  // Lunch options
  { name:'Grilled Chicken Salad',  category:'lunch',     diabeticFriendly:true,  calories:420, fat:12, satFat:2, chol:85, sodium:480, carbs:28, fiber:7, sugar:6,  protein:48 },
  { name:'Quinoa & Veggie Bowl',   category:'lunch',     diabeticFriendly:true,  calories:480, fat:14, satFat:2, chol:0,  sodium:340, carbs:70, fiber:9, sugar:8,  protein:16 },
  { name:'Lentil Soup',            category:'lunch',     diabeticFriendly:true,  calories:380, fat:6,  satFat:1, chol:0,  sodium:520, carbs:60, fiber:14,sugar:5,  protein:20 },
  { name:'Tuna Whole Grain Wrap',  category:'lunch',     diabeticFriendly:true,  calories:440, fat:10, satFat:2, chol:40, sodium:580, carbs:54, fiber:6, sugar:4,  protein:32 },
  { name:'Brown Rice & Stir-Fry',  category:'lunch',     diabeticFriendly:true,  calories:500, fat:10, satFat:2, chol:0,  sodium:620, carbs:82, fiber:6, sugar:6,  protein:18 },
  { name:'Beef Burger & Fries',    category:'lunch',     diabeticFriendly:false, calories:880, fat:42, satFat:14,chol:95, sodium:1100,carbs:90, fiber:4, sugar:12, protein:38 },
  // Dinner options
  { name:'Baked Salmon & Broccoli',category:'dinner',    diabeticFriendly:true,  calories:460, fat:18, satFat:3, chol:100,sodium:360, carbs:22, fiber:5, sugar:4,  protein:54 },
  { name:'Steamed Veggies & Tofu', category:'dinner',    diabeticFriendly:true,  calories:360, fat:10, satFat:1, chol:0,  sodium:420, carbs:42, fiber:8, sugar:8,  protein:22 },
  { name:'Turkey & Sweet Potato',  category:'dinner',    diabeticFriendly:true,  calories:520, fat:8,  satFat:2, chol:85, sodium:480, carbs:64, fiber:7, sugar:12, protein:44 },
  { name:'Grilled Tilapia & Rice', category:'dinner',    diabeticFriendly:true,  calories:490, fat:8,  satFat:2, chol:90, sodium:400, carbs:62, fiber:3, sugar:2,  protein:42 },
  { name:'Vegetable Pasta',        category:'dinner',    diabeticFriendly:true,  calories:540, fat:12, satFat:2, chol:5,  sodium:380, carbs:86, fiber:8, sugar:10, protein:16 },
  { name:'Fried Rice with Egg',    category:'dinner',    diabeticFriendly:false, calories:620, fat:18, satFat:4, chol:185,sodium:780, carbs:90, fiber:3, sugar:4,  protein:18 },
];

// ── DOM Ready ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initSteppers();
  initActivitySlider();
  initPredictForm();
  initFoodRecommendation();
  initImageUpload();
});

// ════════════════════════════════════════════════════════════════════════════
//  NAVIGATION
// ════════════════════════════════════════════════════════════════════════════
function initNavigation() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => switchPage(btn.dataset.page));
  });
}

function switchPage(pageName) {
  STATE.page = pageName;

  // Update nav buttons
  document.querySelectorAll('.nav-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.page === pageName);
  });

  // Show correct section
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  const target = document.getElementById(`page-${pageName}`);
  if (target) target.classList.add('active');

  // Populate health cards if navigating to pages that need them
  if (pageName === 'recommendation') buildRecommendationPage();
  if (pageName === 'upload')          buildUploadPage();

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ════════════════════════════════════════════════════════════════════════════
//  STEPPER BUTTONS
// ════════════════════════════════════════════════════════════════════════════
function initSteppers() {
  document.querySelectorAll('.stepper').forEach(btn => {
    btn.addEventListener('click', () => {
      const input = document.getElementById(btn.dataset.target);
      const step  = parseFloat(btn.dataset.step || 1);
      let val     = parseFloat(input.value) || 0;
      val = btn.classList.contains('inc') ? val + step : val - step;
      val = Math.max(parseFloat(input.min || 0), Math.min(parseFloat(input.max || 9999), val));
      input.value = step < 1 ? val.toFixed(1) : val;
      input.dispatchEvent(new Event('input'));
    });
  });
}

// ════════════════════════════════════════════════════════════════════════════
//  ACTIVITY SLIDER
// ════════════════════════════════════════════════════════════════════════════
function initActivitySlider() {
  const slider = document.getElementById('activity');
  const display = document.getElementById('activity-display');
  slider.addEventListener('input', () => {
    display.textContent = ACTIVITY_LABELS[slider.value];
  });
}

// ════════════════════════════════════════════════════════════════════════════
//  PREDICTION FORM
// ════════════════════════════════════════════════════════════════════════════
function initPredictForm() {
  document.getElementById('predict-form').addEventListener('submit', async e => {
    e.preventDefault();
    if (!validatePredictForm()) return;

    collectPredictInputs();

    const btn  = document.getElementById('predict-btn');
    setButtonLoading(btn, true);

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          glucose:       STATE.glucose,
          bloodPressure: STATE.bloodPressure,
          age:           STATE.age,
          weight:        STATE.weight,
          height:        STATE.height,
          gender:        STATE.gender,
          activity:      STATE.activity,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Prediction failed.');

      // Store results in STATE
      STATE.isDiabetic  = data.isDiabetic;
      STATE.bmi         = data.bmi;
      STATE.bmiCategory = data.bmiCategory;
      STATE.bmiColor    = data.bmiColor;
      STATE.daily       = data.daily;
      STATE.bmr1        = data.bmr1;
      STATE.bmr2        = data.bmr2;
      STATE.bmr3        = data.bmr3;
      STATE.glucose     = data.glucose;
      STATE.healthRows  = data.healthRows;
      STATE.adviceLines = data.adviceLines;
      STATE.conclusion  = data.conclusion;
      STATE.dataString  = data.dataString;

      renderPredictionResults();
      document.getElementById('prediction-results').classList.remove('hidden');
      document.getElementById('prediction-results').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      setButtonLoading(btn, false);
    }
  });
}

function validatePredictForm() {
  const h = parseFloat(document.getElementById('height').value);
  if (h <= 0) { alert('Please enter a valid Height greater than 0.'); return false; }
  const w = parseFloat(document.getElementById('weight').value);
  if (w <= 0) { alert('Please enter a valid Weight greater than 0.'); return false; }
  return true;
}

function collectPredictInputs() {
  STATE.glucose       = parseFloat(document.getElementById('glucose').value)       || 0;
  STATE.bloodPressure = parseFloat(document.getElementById('bloodpressure').value) || 0;
  STATE.age           = parseFloat(document.getElementById('age').value)           || 0;
  STATE.weight        = parseFloat(document.getElementById('weight').value)        || 0;
  STATE.height        = parseFloat(document.getElementById('height').value)        || 0;
  STATE.gender        = document.getElementById('gender').value;
  const idx           = parseInt(document.getElementById('activity').value);
  STATE.activity      = ACTIVITY_LABELS[idx];
  STATE.activityMult  = ACTIVITY_MULTIPLIERS[idx];
}

function runCalculations() {
  const { weight, height, age, gender, activityMult } = STATE;
  const hm = height / 100;

  // BMI
  STATE.bmi = +(weight / (hm * hm)).toFixed(2);

  // BMR — Harris-Benedict
  const bmrRaw = gender === 'Male'
    ? 66.5 + 13.75 * weight + 5.003 * height - 6.75 * age
    : 655.1 + 9.563 * weight + 1.850 * height - 4.676 * age;
  STATE.bmr  = Math.round(bmrRaw);
  STATE.daily = Math.round(bmrRaw * activityMult);

  STATE.bmr1 = Math.round(0.35 * STATE.daily);
  STATE.bmr2 = Math.round(0.40 * STATE.daily);
  STATE.bmr3 = Math.round(0.25 * STATE.daily);

  // BMI category
  if      (STATE.bmi < 18.5) { STATE.bmiCategory = 'Underweight'; STATE.bmiColor = '#ff2b47'; }
  else if (STATE.bmi < 25)   { STATE.bmiCategory = 'Normal';      STATE.bmiColor = '#3cb371'; }
  else if (STATE.bmi < 30)   { STATE.bmiCategory = 'Overweight';  STATE.bmiColor = '#ffa500'; }
  else                        { STATE.bmiCategory = 'Obese';       STATE.bmiColor = '#ff2b47'; }

  // Simple diabetes risk estimation (thresholds from clinical guidelines)
  const risk =
    (STATE.glucose > 140 ? 2 : STATE.glucose > 100 ? 1 : 0) +
    (STATE.bmi > 30 ? 2 : STATE.bmi > 25 ? 1 : 0) +
    (STATE.bloodPressure > 90 ? 1 : 0) +
    (STATE.age > 45 ? 1 : 0);
  STATE.isDiabetic = risk >= 3;

  // Build healthRows and advice text
  buildAdviceText();
}

function buildAdviceText() {
  const { glucose, bloodPressure, bmi, daily, age, gender, activity, isDiabetic, bmiCategory } = STATE;
  const status   = isDiabetic ? 'has diabetes risk' : 'does not have diabetes risk';
  const glucoseNote = glucose > 140 ? 'elevated — indicates potential insulin resistance'
    : glucose > 100 ? 'borderline — requires monitoring'
    : 'within normal range';
  const bpNote = bloodPressure > 90 ? 'elevated — increases cardiovascular risk'
    : bloodPressure > 80 ? 'borderline high'
    : 'within normal diastolic range';
  const bmiNote = bmi >= 30 ? `${bmiCategory} — significantly increases diabetes risk`
    : bmi >= 25 ? `${bmiCategory} — moderately increases diabetes risk`
    : `${bmiCategory} — healthy body weight`;

  STATE.healthRows = [
    ['Glucose Level',              `${glucose} mg/dL — ${glucoseNote}`],
    ['Diastolic Blood Pressure',   `${bloodPressure} mmHg — ${bpNote}`],
    ['BMI',                        `${STATE.bmi} kg/m² — ${bmiNote}`],
    ['Daily Calories (BMR)',       `${daily} kcal/day based on ${activity}`],
    ['Age',                        `${age} years old`],
    ['Gender',                     gender],
    ['Diabetes Status',            `Patient ${status}`],
  ];

  const advice = [];
  if (glucose > 140)        advice.push('Reduce intake of refined sugars and high-glycemic foods to help lower blood sugar levels.');
  if (glucose > 100)        advice.push('Monitor blood sugar regularly and consider a low-glycemic diet rich in vegetables and whole grains.');
  if (bloodPressure > 90)   advice.push('Reduce sodium intake to below 2,300 mg/day and practice stress-management techniques.');
  if (bmi >= 30)            advice.push('Aim for a gradual 5–10% body weight reduction through a combination of diet and moderate exercise.');
  else if (bmi >= 25)       advice.push('Include 30 minutes of moderate aerobic activity (brisk walking, cycling) at least 5 days per week.');
  if (age > 45)             advice.push('Schedule regular health screenings every 6–12 months, including HbA1c and lipid panel tests.');
  advice.push('Stay well-hydrated — aim for 8 glasses (2 litres) of water per day.');
  advice.push('Prioritize 7–9 hours of quality sleep each night to support metabolic health.');
  advice.push('Incorporate stress-reduction practices such as meditation, yoga, or light stretching.');
  advice.push(`Your estimated daily calorie target is ${daily} kcal — distributed as: Breakfast ${STATE.bmr1} kcal, Lunch ${STATE.bmr2} kcal, Dinner ${STATE.bmr3} kcal.`);

  STATE.adviceLines = advice;

  STATE.conclusion = isDiabetic
    ? 'Based on your health data, you show indicators associated with diabetes risk. Please consult a healthcare professional for a comprehensive clinical assessment.'
    : 'Based on your health data, your current indicators do not suggest active diabetes. Continue maintaining a healthy lifestyle to preserve this status.';

  STATE.dataString = `glucose level: ${glucose} mg/dL, diastolic blood pressure: ${bloodPressure} mmHg, BMI: ${STATE.bmi} kg/m², daily calories: ${daily} kcal, age: ${age}, gender: ${gender}, activity: ${activity}, diabetes status: ${status}`;
}

function renderPredictionResults() {
  const { isDiabetic, bmi, daily, bmr1, bmr2, bmr3, bmiCategory, bmiColor } = STATE;
  const color  = isDiabetic ? '#ff2b47' : '#3cb371';
  const label  = isDiabetic ? 'Patient has diabetes risk' : 'Patient does not have diabetes risk';

  // Status box
  const statusBox = document.getElementById('diabetes-status-box');
  statusBox.className = `status-box ${isDiabetic ? 'negative' : 'positive'}`;
  statusBox.innerHTML = `<span class="status-dot"></span>${isDiabetic ? '⚠️ Patient shows diabetes risk indicators' : '✅ Patient does not show diabetes risk indicators'}`;

  // Metric cards
  setMetricCard('bmi-card',         `${bmi} kg/m²`,           'Body Mass Index (BMI)');
  setMetricCard('bmr-card',         `${daily} Calories/day`,  'Daily Calories (BMR)');
  setMetricCard('breakfast-cal-card', `${bmr1} Cal`,           'Breakfast (Calories)');
  setMetricCard('lunch-cal-card',     `${bmr2} Cal`,           'Lunch (Calories)');
  setMetricCard('dinner-cal-card',    `${bmr3} Cal`,           'Dinner (Calories)');

  // BMI badge
  const badge = document.getElementById('bmi-category-badge');
  const bmiClass = { 'Underweight':'underweight', 'Normal':'normal', 'Overweight':'overweight', 'Obese':'obese' }[bmiCategory] || 'normal';
  badge.className = `bmi-badge ${bmiClass}`;
  badge.innerHTML = `<strong>${bmiCategory} — ${STATE.bmi} kg/m²</strong><p>A healthy BMI is generally between 18.5 – 25 kg/m²</p>`;

  // Health details table
  document.getElementById('health-details-table').innerHTML = buildTable(
    ['Health Data', 'Description'], STATE.healthRows
  );

  // Advice
  const adviceEl = document.getElementById('lifestyle-advice');
  adviceEl.innerHTML = '<ul>' + STATE.adviceLines.map(a => `<li>${a}</li>`).join('') + '</ul>';

  // Conclusion
  const conclBox = document.getElementById('conclusion-box');
  conclBox.className = `conclusion-box ${isDiabetic ? 'negative' : 'positive'}`;
  conclBox.textContent = STATE.conclusion;
}

// ════════════════════════════════════════════════════════════════════════════
//  FOOD RECOMMENDATION PAGE
// ════════════════════════════════════════════════════════════════════════════
function buildRecommendationPage() {
  const hasPrediction = STATE.daily !== null;
  const warning = document.getElementById('rec-prediction-warning');
  const gated   = document.getElementById('rec-gated-content');

  if (!hasPrediction) {
    warning.classList.remove('hidden');
    gated.classList.add('hidden');
    return;
  }

  warning.classList.add('hidden');
  gated.classList.remove('hidden');

  buildUploadHealthCards('rec-health-cards');
  buildFoodGrid();

  document.getElementById('check-nutrition-btn').onclick = async () => {
    const btn = document.getElementById('check-nutrition-btn');
    setButtonLoading(btn, true);
    try {
      await renderFoodAnalysis();
    } finally {
      setButtonLoading(btn, false);
    }
  };
}

async function buildFoodGrid() {
  const grid = document.getElementById('food-grid');
  grid.innerHTML = '<p style="text-align:center">Loading recommendations…</p>';

  try {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ daily: STATE.daily, bmr1: STATE.bmr1, bmr2: STATE.bmr2, bmr3: STATE.bmr3 }),
    });
    const data = await res.json();

    // Map API response into FOOD_DB-compatible entries
    // Robust mapper: accept different casing/column names from API/CSV (e.g. Kalori vs kalori, Lemak vs lemak)
    const mapMeal = (records, category) => (records || []).map(r => {
      const get = (...keys) => {
        for (const k of keys) if (r[k] !== undefined && r[k] !== null) return r[k];
        return undefined;
      };
      const parseNum = v => {
        if (v === undefined || v === null) return 0;
        if (typeof v === 'string') v = v.replace(/,/g, '.').replace(/[^0-9.\-]/g, '');
        const n = parseFloat(v);
        return Number.isFinite(n) ? n : 0;
      };

      return {
        name:             get('makanan', 'nama_makanan', 'nama', 'name') || 'Unknown',
        category,
        diabeticFriendly: !!(get('diabetic_friendly', 'diabeticFriendly', 'diabetic') || false),
        calories:         parseNum(get('Kalori', 'kalori')),
        fat:              parseNum(get('Lemak', 'lemak')),
        satFat:           parseNum(get('lemakJenuh', 'lemak_jenuh', 'lemakJenuh')) || 0,
        chol:             parseNum(get('kolesterol', 'Kolesterol', 'cholesterol')) || 0,
        sodium:           parseNum(get('sodium', 'Natrium', 'natrium')),
        carbs:            parseNum(get('karbohidrat', 'Karbohidrat')),
        fiber:            parseNum(get('serat', 'seratpangan', 'SeratPangan')),
        sugar:            parseNum(get('gula', 'Gula')),
        protein:          parseNum(get('protein', 'Protein')),
      };
    });

    STATE.recommended.breakfast = mapMeal(data.breakfast, 'breakfast');
    STATE.recommended.lunch     = mapMeal(data.lunch,     'lunch');
    STATE.recommended.dinner    = mapMeal(data.dinner,    'dinner');

    // Debug: log raw API rows and mapped results to help diagnose missing values
    try {
      console.debug('API raw breakfast rows:', data.breakfast);
      console.debug('Mapped breakfast:', STATE.recommended.breakfast);
      console.debug('API raw lunch rows:', data.lunch);
      console.debug('Mapped lunch:', STATE.recommended.lunch);
      console.debug('API raw dinner rows:', data.dinner);
      console.debug('Mapped dinner:', STATE.recommended.dinner);
    } catch (e) { /* ignore logging errors */ }
  } catch {
    // Fallback to hardcoded FOOD_DB if API fails
    ['breakfast', 'lunch', 'dinner'].forEach(cat => {
      STATE.recommended[cat] = FOOD_DB.filter(f => f.category === cat && (!STATE.isDiabetic || f.diabeticFriendly));
    });
  }

  grid.innerHTML = '';
  const categories = ['breakfast', 'lunch', 'dinner'];
  const labels     = ['🌅 Breakfast Menu', '🌤️ Lunch Menu', '🌙 Dinner Menu'];

  categories.forEach((cat, i) => {
    const items = STATE.recommended[cat];
    const col = document.createElement('div');
    col.className = 'food-column';
    col.innerHTML = `<h4>${labels[i]}</h4>`;

    items.forEach(food => {
      const exp = document.createElement('div');
      exp.className = 'food-item-expander';
      exp.innerHTML = `
        <div class="food-item-header" onclick="toggleFoodItem(this)">
          <span>${food.name}</span><span>▼</span>
        </div>
        <div class="food-item-body">
          ${buildNutritionTable(food)}
        </div>`;
      col.appendChild(exp);
    });
    grid.appendChild(col);
  });

  populateMealSelects();
}

function toggleFoodItem(header) {
  const body = header.nextElementSibling;
  const icon = header.querySelector('span:last-child');
  const open = body.classList.toggle('open');
  icon.style.transform = open ? 'rotate(180deg)' : '';
}

function buildNutritionTable(food) {
  const rows = [
    ['Calories (kcal)', food.calories],
    ['Fat (g)',         food.fat],
    ['Saturated Fat (g)', food.satFat],
    ['Cholesterol (mg)', food.chol],
    ['Sodium (mg)',     food.sodium],
    ['Carbohydrates (g)', food.carbs],
    ['Fiber (g)',       food.fiber],
    ['Sugar (g)',       food.sugar],
    ['Protein (g)',     food.protein],
  ];
  return buildTable(['Nutrient', 'Amount'], rows);
}

function populateMealSelects() {
  ['breakfast', 'lunch', 'dinner'].forEach(cat => {
    const sel = document.getElementById(`sel-${cat}`);
    sel.innerHTML = '';
    STATE.recommended[cat].forEach(f => {
      const opt = new Option(f.name, f.name);
      sel.appendChild(opt);
    });
  });
}

async function renderFoodAnalysis() {

  const breakfastName = document.getElementById('sel-breakfast').value;
  const lunchName     = document.getElementById('sel-lunch').value;
  const dinnerName    = document.getElementById('sel-dinner').value;

  const bf = STATE.recommended.breakfast.find(f => f.name === breakfastName);
  const lu = STATE.recommended.lunch.find(f => f.name === lunchName);
  const dn = STATE.recommended.dinner.find(f => f.name === dinnerName);

  const totalCal = (bf?.calories || 0) + (lu?.calories || 0) + (dn?.calories || 0);
  const diff     = totalCal - STATE.daily;
  const diffText = diff > 0
    ? `${Math.abs(diff)} kcal above your daily target`
    : diff < 0
    ? `${Math.abs(diff)} kcal below your daily target`
    : 'exactly at your daily calorie target';

  const concl = document.getElementById('food-conclusion-box');
  concl.className = 'conclusion-box neutral';

  // Fetch AI reasoning — keep spinner going until this resolves
  let aiOk = false;
  try {
    const res  = await fetch('/api/food-analysis', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        breakfast:  bf  || null,
        lunch:      lu  || null,
        dinner:     dn  || null,
        healthData: STATE.dataString || '',
      }),
    });
    const data = await res.json();
    if (data.ok && Array.isArray(data.sections) && data.sections.length >= 3) {
      renderMealBlockAI('breakfast', '🌅 Breakfast menu', bf, STATE.bmr1, data.sections[0]);
      renderMealBlockAI('lunch',     '🌤️ Lunch menu',     lu, STATE.bmr2, data.sections[1]);
      renderMealBlockAI('dinner',    '🌙 Dinner menu',    dn, STATE.bmr3, data.sections[2]);
      // Update conclusion from AI if available
      const aiConcl = (data.sections[3] || '').replace(/^Conclusion:\s*/i, '').trim();
      concl.textContent = aiConcl ||
        `Your selected meals total ${totalCal} kcal, which is ${diffText} of ${STATE.daily} kcal.`;
      aiOk = true;
    }
  } catch (e) { /* fall through to static */ }

  // Only use static rendering if AI failed
  if (!aiOk) {
    renderMealBlock('breakfast', '🌅 Breakfast menu', bf, STATE.bmr1);
    renderMealBlock('lunch',     '🌤️ Lunch menu',     lu, STATE.bmr2);
    renderMealBlock('dinner',    '🌙 Dinner menu',    dn, STATE.bmr3);
    concl.textContent =
      `Your selected meals total ${totalCal} kcal, which is ${diffText} of ${STATE.daily} kcal. ` +
      (STATE.isDiabetic
        ? 'All recommended meals are diabetic-friendly. Monitor your portions and blood sugar after meals.'
        : 'Great choices! Maintain variety and consistency for optimal health outcomes.');
  }

  document.getElementById('food-analysis-results').classList.remove('hidden');
  document.getElementById('food-analysis-results').scrollIntoView({ behavior: 'smooth' });
}

function renderMealBlockAI(meal, icon, food, targetCal, aiText) {
  const titleEl = document.getElementById(`${meal}-meal-title`);
  const tableEl = document.getElementById(`${meal}-meal-table`);

  const lines = (aiText || '').split('\n').map(l => l.trim()).filter(Boolean);
  // First line is the meal header, e.g. "Breakfast menu: Ketoprak..."
  titleEl.textContent = lines[0] || (food ? `${icon}: ${food.name}` : icon);

  // Parse lines that start with "-" as key: value rows
  const rows = lines
    .filter(l => l.startsWith('-'))
    .map(l => {
      const clean = l.replace(/^-\s*/, '');
      const colon = clean.indexOf(':');
      if (colon === -1) return [clean, ''];
      return [clean.slice(0, colon).trim(), clean.slice(colon + 1).trim()];
    })
    .filter(r => r[0]);

  if (rows.length) {
    tableEl.innerHTML = buildTable(['Nutrition Info', 'Description'], rows);
  } else {
    // Nothing to parse — keep the static fallback
    renderMealBlock(meal, icon, food, targetCal);
  }
}

function renderMealBlock(meal, icon, food, targetCal) {
  const titleEl = document.getElementById(`${meal}-meal-title`);
  const tableEl = document.getElementById(`${meal}-meal-table`);

  titleEl.textContent = `${icon}: ${food.name}`;
  const diff    = food.calories - targetCal;
  const pattern = Math.abs(diff) < 60 ? 'On target' : diff > 0 ? `${diff} kcal over target` : `${Math.abs(diff)} kcal under target`;

  const rows = [
    ['Eating pattern',  food.diabeticFriendly ? 'Diabetic-friendly, balanced macros' : 'Standard — monitor sugar intake'],
    ['Sugar content',   `${food.sugar} g`],
    ['Daily calories',  `${food.calories} kcal (target: ${targetCal} kcal — ${pattern})`],
    ['Nutrition',       `Protein: ${food.protein}g | Carbs: ${food.carbs}g | Fat: ${food.fat}g | Fiber: ${food.fiber}g`],
  ];
  tableEl.innerHTML = buildTable(['Nutrition Info', 'Description'], rows);
}

// ════════════════════════════════════════════════════════════════════════════
//  UPLOAD / IMAGE ANALYSIS PAGE
// ════════════════════════════════════════════════════════════════════════════
function buildUploadPage() {
  const hasPrediction = STATE.daily !== null;
  const warning = document.getElementById('upload-prediction-warning');
  const gated   = document.getElementById('upload-gated-content');

  // Show gated content and warning based on prediction state
  if (warning) warning.classList.toggle('hidden', hasPrediction);
  if (gated)   gated.classList.toggle('hidden', !hasPrediction);

  if (hasPrediction) buildUploadHealthCards('upload-health-cards');
}

function buildUploadHealthCards(containerId) {
  const container = document.getElementById(containerId);
  if (!container || !STATE.daily) {
    if (container) container.innerHTML =
      `<div class="info-box">⚠️ Please complete the Prediction page first to populate your health data.</div>`;
    return;
  }

  container.innerHTML = `
    <div class="cards-row-2">
      ${metricCardHTML(`${STATE.glucose} mg/dL`,  'Blood Sugar Level')}
      ${metricCardHTML(`${STATE.daily} Cal/day`,   'Daily Calories (BMR)')}
    </div>
    <div class="cards-row-3">
      ${metricCardHTML(`${STATE.bmr1} Cal`, 'Breakfast (Calories)')}
      ${metricCardHTML(`${STATE.bmr2} Cal`, 'Lunch (Calories)')}
      ${metricCardHTML(`${STATE.bmr3} Cal`, 'Dinner (Calories)')}
    </div>`;
}

function _applyImageToUI(dataUrl, fileName) {
  document.getElementById('upload-zone').style.display    = 'none';
  document.getElementById('upload-preview').style.display = 'flex';
  document.getElementById('preview-img').src              = dataUrl;
  document.getElementById('preview-filename').textContent = fileName;
  // keep analysis results hidden until Analyze is clicked
  document.getElementById('image-analysis-results').classList.add('hidden');
  document.getElementById('upload-info').classList.remove('hidden');
}

function initImageUpload() {
  var fileInput = document.getElementById('file-input');
  if (!fileInput) return;

  // File selection is handled by inline onchange in index.html
  // (sets STATE.uploadedImage synchronously before FileReader runs)

  // Drag-and-drop
  var zone = document.getElementById('upload-zone');
  zone.addEventListener('dragover',  function (e) { e.preventDefault(); zone.style.borderColor = 'var(--teal)'; });
  zone.addEventListener('dragleave', function ()  { zone.style.borderColor = ''; });
  zone.addEventListener('drop', function (e) {
    e.preventDefault();
    zone.style.borderColor = '';
    var file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    STATE.uploadedImage = file;
    var reader = new FileReader();
    reader.onload = function (ev) {
      STATE.previewUrl = ev.target.result;
      _applyImageToUI(ev.target.result, file.name);
    };
    reader.readAsDataURL(file);
  });

  // Analyse and Reset are wired via onclick in HTML (see analyzeImage / resetImageUpload globals)
}

// ── Global handlers called directly from onclick attributes in HTML ─────────
async function analyzeImage() {
  var btn = document.getElementById('analyse-image-btn');
  if (!STATE.uploadedImage) { alert('Please upload an image first.'); return; }
  setButtonLoading(btn, true);
  try {
    var form = new FormData();
    form.append('image', STATE.uploadedImage);
    form.append('healthData', STATE.dataString || '');
    var res  = await fetch('/api/analyze', { method: 'POST', body: form });
    var data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Analysis failed.');
    renderImageAnalysis(data);
    document.getElementById('image-analysis-results').classList.remove('hidden');
    document.getElementById('image-analysis-results').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    setButtonLoading(btn, false);
  }
}

async function loadSampleImage() {
  try {
    var res  = await fetch('/api/sample-image');
    if (!res.ok) throw new Error('Could not load sample image.');
    var blob = await res.blob();
    var fileName = 'Food picture composition for analysis.png';
    var file = new File([blob], fileName, { type: 'image/png' });
    STATE.uploadedImage = file;
    var reader = new FileReader();
    reader.onload = function(e) {
      STATE.previewUrl = e.target.result;
      document.getElementById('preview-img').src = e.target.result;
      document.getElementById('preview-filename').textContent = fileName;
      document.getElementById('upload-zone').style.display    = 'none';
      document.getElementById('upload-preview').style.display = 'flex';
      document.getElementById('image-analysis-results').classList.add('hidden');
      document.getElementById('upload-info').classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  } catch (err) {
    alert('Error loading sample image: ' + err.message);
  }
}

function resetImageUpload() {
  var fileInput = document.getElementById('file-input');
  STATE.uploadedImage = null;
  STATE.previewUrl    = null;
  if (fileInput) fileInput.value = '';
  document.getElementById('preview-img').src              = '';
  document.getElementById('preview-filename').textContent = '';
  document.getElementById('upload-preview').style.display = 'none';
  document.getElementById('upload-zone').style.display    = '';
  var resultsEl = document.getElementById('image-analysis-results');
  resultsEl.classList.add('hidden');
  ['product-type-banner','nutritional-content-table','product-recommendation-box',
   'detailed-reasons','nutrition-info-table','image-conclusion-box'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.innerHTML = '';
  });
  document.getElementById('upload-info').classList.remove('hidden');
}

function renderImageAnalysis(apiData) {

  const file = STATE.uploadedImage;
  const url  = STATE.previewUrl || document.getElementById('preview-img').src;

  // Parse AI analysis text
  const analysisText = apiData.analysis || '';
  const composition  = apiData.composition || '';

  // uploaded image display removed — analysis shows only parsed results

  // Product type from composition OCR
  const productTypeMatch = composition.match(/Product type:\s*(.+)/i);
  const productType = productTypeMatch ? productTypeMatch[1].trim() : file.name.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ');
  document.getElementById('product-type-banner').innerHTML =
    `🍔 Product: <strong>${productType}</strong>`;

  // Parse nutritional lines from composition OCR
  const compLines = composition.split('\n').filter(l => /^\d+\./.test(l.trim()));
  const nutritionRows = compLines.map(l => {
    const parts = l.replace(/^\d+\.\s*/, '').split(':');
    return [parts[0]?.trim() || '', parts.slice(1).join(':').trim() || ''];
  });
  if (nutritionRows.length === 0) {
    nutritionRows.push(['See raw composition below', composition]);
  }
  document.getElementById('nutritional-content-table').innerHTML =
    buildTable(['Nutritional Content', 'Amount'], nutritionRows);

  // Recommendation from analysis text
  const recMatch = analysisText.match(/Is the product recommended\?:\s*(yes|no)/i);
  const isSafe   = recMatch ? recMatch[1].toLowerCase() === 'yes' : !STATE.isDiabetic;
  const recBox   = document.getElementById('product-recommendation-box');
  recBox.className = `status-box ${isSafe ? 'positive' : 'negative'}`;
  recBox.innerHTML = `<span class="status-dot"></span>${isSafe ? '✅ Product is recommended' : '⚠️ Product is not recommended'}`;

  // Detailed reasons
  const reasonsSection = analysisText.match(/Detailed reasons:([\s\S]*?)(?=Nutrition information:|Conclusion:|$)/i);
  const rawReasons = reasonsSection ? reasonsSection[1].trim() : '';
  const reasons = rawReasons.split('\n').map(l => l.replace(/^-\s*/, '').trim()).filter(Boolean);
  document.getElementById('detailed-reasons').innerHTML =
    '<ul>' + (reasons.length ? reasons : ['See full analysis below.']).map(r => `<li>${r}</li>`).join('') + '</ul>';

  // Nutrition info rows
  const infoSection = analysisText.match(/Nutrition information:([\s\S]*?)(?=Conclusion:|$)/i);
  const rawInfo = infoSection ? infoSection[1].trim() : '';
  const infoRows = rawInfo.split('\n').map(l => {
    const parts = l.replace(/^-\s*/, '').split(':');
    return [parts[0]?.trim(), parts.slice(1).join(':').trim()];
  }).filter(r => r[0] && r[1]);
  if (infoRows.length === 0) infoRows.push(['Analysis', analysisText]);
  document.getElementById('nutrition-info-table').innerHTML =
    buildTable(['Nutrition Info', 'Description'], infoRows);

  // Conclusion
  const conclMatch = analysisText.match(/Conclusion:([\s\S]*$)/i);
  const conclusionText = conclMatch ? conclMatch[1].trim() : (isSafe
    ? `"${productType}" appears suitable for your health profile. Consume in moderation.`
    : `"${productType}" is not recommended given your current health indicators.`);
  const concl = document.getElementById('image-conclusion-box');
  concl.className = `conclusion-box ${isSafe ? 'positive' : 'negative'}`;
  concl.textContent = conclusionText;
}

// ════════════════════════════════════════════════════════════════════════════
//  HELPERS / UTILITIES
// ════════════════════════════════════════════════════════════════════════════
function setMetricCard(id, value, label) {
  document.getElementById(id).innerHTML =
    `<div class="card-label">${label}</div><div class="card-value">${value}</div>`;
}

function metricCardHTML(value, label) {
  return `<div class="metric-card">
    <div class="card-label">${label}</div>
    <div class="card-value">${value}</div>
  </div>`;
}

function buildTable(headers, rows) {
  const ths = headers.map(h => `<th>${h}</th>`).join('');
  const trs = rows.map(row =>
    '<tr>' + row.map(cell => `<td>${cell}</td>`).join('') + '</tr>'
  ).join('');
  return `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
}

function toggleExpander(header) {
  const body  = header.nextElementSibling;
  const chev  = header.querySelector('.chevron');
  const open  = body.classList.toggle('collapsed');
  chev.style.transform = open ? 'rotate(-90deg)' : '';
}

function setButtonLoading(btn, loading) {
  btn.disabled = loading;
  btn.querySelector('.btn-text').classList.toggle('hidden',    loading);
  btn.querySelector('.btn-spinner').classList.toggle('hidden', !loading);
}

function simulateDelay(ms) {
  return new Promise(res => setTimeout(res, ms));
}
