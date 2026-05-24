import axios from 'axios';

const API_BASE = 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

export const predictHeartDisease = async (formData) => {
  const response = await api.post('/predict', {
    age: parseInt(formData.age),
    sex: formData.sex,
    chestPainType: formData.chestPainType,
    restingBP: parseInt(formData.restingBP),
    cholesterol: parseInt(formData.cholesterol),
    fastingBS: parseInt(formData.fastingBS),
    restingECG: formData.restingECG,
    maxHR: parseInt(formData.maxHR),
    exerciseAngina: formData.exerciseAngina,
    oldpeak: parseFloat(formData.oldpeak),
    stSlope: formData.stSlope,
  });
  return response.data;
};

export const getModelInfo = async () => {
  const response = await api.get('/model-info');
  return response.data;
};

export default api;