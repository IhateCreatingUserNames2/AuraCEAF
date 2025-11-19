// Aura CEAF V3 Frontend - Refactored Script
// ==========================================

// Configuration and global state
const CONFIG = {
    API_BASE: window.location.origin + '/ceaf',
};

let drivesChart = null;

const state = {
    currentUser: null,
    isAuthenticated: false,
    currentAgent: null,
    sessionId: null,
    currentView: 'discover',
    agents: [], // User's own agents
    messages: [],
    isTyping: false,
};

// DOM elements cache
const elements = {};

// ==========================================
// INITIALIZATION
// ==========================================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Aura CEAF V3 Frontend Initialized');
    cacheDOMElements();
    setupEventListeners();
    await checkAuthentication();
    initializeUI();
    initializeDrivesChart();
});

function cacheDOMElements() {
    // Main layout
    elements.sidebar = document.getElementById('sidebar');
    elements.mainContent = document.getElementById('main-content');
    elements.loadingOverlay = document.getElementById('loading-overlay');

    // Sidebar
    elements.sidebarToggle = document.getElementById('sidebar-toggle');
    elements.agentsList = document.getElementById('agents-list');
    elements.agentCount = document.getElementById('agent-count');

    // User Profile & Menu
    elements.userProfile = document.getElementById('user-profile');
    elements.userMenu = document.getElementById('user-menu');
    elements.usernameDisplay = document.getElementById('username-display');
    elements.userInitial = document.getElementById('user-initial');
    elements.loginBtn = document.getElementById('login-btn');
    elements.logoutBtn = document.getElementById('logout-btn');

    // Navigation
    elements.discoverTab = document.getElementById('discover-tab');
    elements.myAgentsTab = document.getElementById('my-agents-tab');
    elements.createAgentNavBtn = document.getElementById('create-agent-nav-btn');

    // Views
    elements.discoverView = document.getElementById('discover-view');
    elements.myAgentsView = document.getElementById('my-agents-view');
    elements.chatView = document.getElementById('chat-view');
    elements.createView = document.getElementById('create-view');

    // Discover / Marketplace
    elements.featuredGrid = document.getElementById('featured-grid');

    // My Agents
    elements.myAgentsGrid = document.getElementById('my-agents-grid');

    // Chat View
    elements.chatHeader = document.getElementById('chat-header');
    elements.currentAgentAvatar = document.getElementById('current-agent-avatar');
    elements.currentAgentName = document.getElementById('current-agent-name');
    elements.currentAgentDescription = document.getElementById('current-agent-description');
    elements.chatModelSelector = document.getElementById('chat-model-selector');
    elements.menuBtn = document.getElementById('menu-btn');
    elements.agentDropdownMenu = document.getElementById('agent-dropdown-menu');
    elements.chatMessages = document.getElementById('chat-messages');
    elements.messageInput = document.getElementById('message-input');
    elements.sendBtn = document.getElementById('send-btn');

    // Agent Creation
    elements.agentForm = document.getElementById('agent-form');
    elements.modelSelectCreate = document.getElementById('model-select-create');
    elements.createButton = document.getElementById('create-button');

    // Modals
    elements.authModal = document.getElementById('auth-modal');
    elements.authModalClose = document.getElementById('auth-modal-close');
    elements.authForm = document.getElementById('auth-form');
    elements.errorContainer = document.getElementById('error-container');
    elements.emailGroup = document.getElementById('email-group');
    elements.loginTab = document.getElementById('login-tab');
    elements.registerTab = document.getElementById('register-tab');
    elements.authSubmitBtn = document.getElementById('auth-submit-btn');

    elements.filesModal = document.getElementById('files-modal');
    elements.filesListContainer = document.getElementById('files-list-container');
    elements.knowledgeFileInput = document.getElementById('knowledge-file-input');
    elements.knowledgeFileUploadArea = document.getElementById('knowledge-file-upload-area');
    elements.cognitiveDashboard = document.getElementById('cognitive-dashboard');
    elements.toggleDashboardBtn = document.getElementById('toggle-dashboard-btn');
    elements.dashboardCloseBtn = document.getElementById('dashboard-close-btn');
    elements.fatigueBar = document.getElementById('fatigue-bar');
    elements.fatigueValue = document.getElementById('fatigue-value');
    elements.saturationBar = document.getElementById('saturation-bar');
    elements.saturationValue = document.getElementById('saturation-value');
    elements.flowStateLabel = document.getElementById('flow-state-label');
    elements.reasoningSlider = document.getElementById('reasoning-slider');
    elements.cognitiveModeLabel = document.getElementById('cognitive-mode-label');
    elements.identityVersionLabel = document.getElementById('identity-version-label');
}

function setupEventListeners() {
    // Sidebar & User
    elements.sidebarToggle?.addEventListener('click', toggleSidebar);
    elements.userProfile?.addEventListener('click', toggleUserMenu);
    elements.loginBtn?.addEventListener('click', showAuthModal);
    elements.logoutBtn?.addEventListener('click', logout);

    elements.chatModelSelector?.addEventListener('change', handleModelChange);


    // Navigation
    elements.discoverTab?.addEventListener('click', () => switchView('discover'));
    elements.myAgentsTab?.addEventListener('click', () => switchView('my-agents'));
    elements.createAgentNavBtn?.addEventListener('click', () => switchView('create'));

    // Auth Modal
    elements.authModal?.addEventListener('click', (e) => { if (e.target === elements.authModal) hideAuthModal(); });
    elements.authModalClose?.addEventListener('click', hideAuthModal);
    elements.authForm?.addEventListener('submit', handleAuth);

    // Agent Creation
    elements.agentForm?.addEventListener('submit', handleAgentCreation);
    elements.agentForm?.addEventListener('input', validateCreateForm);

    // Chat
    elements.messageInput?.addEventListener('keypress', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
    elements.messageInput?.addEventListener('input', () => { elements.sendBtn.disabled = elements.messageInput.value.trim().length === 0 || state.isTyping; });
    elements.sendBtn?.addEventListener('click', sendMessage);
    elements.menuBtn?.addEventListener('click', toggleAgentDropdown);
    document.getElementById('agent-files-menu-btn')?.addEventListener('click', showFilesModal);

    elements.toggleDashboardBtn?.addEventListener('click', toggleCognitiveDashboard);
    elements.dashboardCloseBtn?.addEventListener('click', toggleCognitiveDashboard);
    // RAG Files Modal
    elements.knowledgeFileUploadArea?.addEventListener('click', () => elements.knowledgeFileInput.click());
    elements.knowledgeFileUploadArea?.addEventListener('dragover', (e) => { e.preventDefault(); e.currentTarget.classList.add('dragover'); });
    elements.knowledgeFileUploadArea?.addEventListener('dragleave', (e) => { e.currentTarget.classList.remove('dragover'); });
    elements.knowledgeFileUploadArea?.addEventListener('drop', (e) => {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            elements.knowledgeFileInput.files = e.dataTransfer.files;
            handleKnowledgeFileUpload({ target: elements.knowledgeFileInput });
        }
    });

    // Global
    document.addEventListener('click', (e) => {
        if (!elements.userProfile?.contains(e.target) && !elements.userMenu?.contains(e.target)) {
            elements.userMenu.style.display = 'none';
        }
        if (!elements.menuBtn?.contains(e.target) && !elements.agentDropdownMenu?.contains(e.target)) {
            elements.agentDropdownMenu.style.display = 'none';
        }
    });
}

function getCssVariable(variable) {
    return getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
}


function initializeUI() {
    switchAuthTab('login');
    updateUIForAuthentication();
    switchView('discover');
}


function updateCognitiveDashboard(telemetry) {
    if (!telemetry) return;

    // 1. Estado Interno
    const fatigue = telemetry.body_cognitive_fatigue * 100;
    const saturation = telemetry.body_info_saturation * 100;

    // Garante que os elementos existem antes de tentar usá-los
    if (elements.fatigueBar) {
        elements.fatigueBar.value = fatigue;
        elements.fatigueBar.classList.toggle('high', fatigue > 70);
    }
    if (elements.fatigueValue) {
        elements.fatigueValue.textContent = `${fatigue.toFixed(0)}%`; // Atualiza o texto
    }

    if (elements.saturationBar) {
        elements.saturationBar.value = saturation;
        elements.saturationBar.classList.toggle('high', saturation > 70);
    }
    if (elements.saturationValue) {
        elements.saturationValue.textContent = `${saturation.toFixed(0)}%`; // Atualiza o texto
    }

    if (elements.flowStateLabel) {
        const flow = telemetry.cognitive_flow;
        const strain = telemetry.cognitive_strain;
        if (flow > 0.6 && strain < 0.4) {
            elements.flowStateLabel.textContent = "Flowing";
            elements.flowStateLabel.style.color = 'var(--accent-color)';
        } else if (strain > 0.6) {
            elements.flowStateLabel.textContent = "Strained";
            elements.flowStateLabel.style.color = 'var(--error-color)';
        } else {
            elements.flowStateLabel.textContent = "Calm";
            elements.flowStateLabel.style.color = 'var(--text-secondary)';
        }
    }

    // 2. Drives Motivacionais
    if (drivesChart) {
        drivesChart.data.datasets[0].data = [
            telemetry.drive_curiosity,
            telemetry.drive_connection,
            telemetry.drive_mastery,
            telemetry.drive_consistency
        ];
        drivesChart.update();
    }

    // 3. Bússola de Raciocínio
    if (elements.reasoningSlider) {
        const totalBias = telemetry.mcl_coherence_bias + telemetry.mcl_novelty_bias;
        const coherencePercentage = totalBias > 0 ? (telemetry.mcl_coherence_bias / totalBias) * 100 : 50;
        elements.reasoningSlider.value = coherencePercentage;
    }
    if (elements.cognitiveModeLabel) {
        elements.cognitiveModeLabel.textContent = telemetry.mcl_cognitive_state.replace('_', ' ');
    }

    // 4. Identidade
    if (elements.identityVersionLabel) {
        elements.identityVersionLabel.textContent = `v${telemetry.identity_version}`;
    }
}


function initializeDrivesChart() {
    // Pega as cores do tema do CSS para injetar no gráfico
    const textColor = getCssVariable('--text-secondary');
    const borderColor = getCssVariable('--border-primary');
    const accentColor = 'rgba(118, 75, 162, 1)';     // Cor principal do gráfico
    const accentColorBg = 'rgba(118, 75, 162, 0.2)'; // Fundo do gráfico

    const ctx = document.getElementById('drives-radar-chart').getContext('2d');
    if (drivesChart) {
        drivesChart.destroy();
    }
    drivesChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Curiosity', 'Connection', 'Mastery', 'Consistency'],
            datasets: [{
                label: 'Drive Level',
                data: [0.5, 0.5, 0.5, 0.5], // Valores iniciais
                backgroundColor: accentColorBg,
                borderColor: accentColor,
                borderWidth: 2,
                pointBackgroundColor: accentColor
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: {
                        display: true, // Exibe as linhas de eixo
                        color: borderColor // Define a cor das linhas de eixo
                    },
                    grid: {
                        color: borderColor // Define a cor das linhas da grade (círculos)
                    },
                    suggestedMin: 0,
                    suggestedMax: 1,
                    pointLabels: {
                        color: textColor, // Define a cor dos labels (Curiosity, etc.)
                        font: {
                            size: 12 // Opcional: aumenta um pouco a fonte
                        }
                    },
                    ticks: {
                        color: textColor, // Define a cor dos números (0.2, 0.4...)
                        backdropColor: 'transparent', // Remove o fundo branco dos números
                        stepSize: 0.2 // Garante uma escala limpa de 0 a 1
                    }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}


// ==========================================
// AUTHENTICATION & USER STATE
// ==========================================
async function checkAuthentication() {
    const token = localStorage.getItem('aura_token');
    if (!token) {
        state.isAuthenticated = false;
        return;
    }
    try {
        const response = await apiRequest('/auth/me', { headers: { 'Authorization': `Bearer ${token}` } });
        state.currentUser = response;
        state.isAuthenticated = true;
    } catch (error) {
        console.warn('Token validation failed:', error);
        localStorage.removeItem('aura_token');
        state.isAuthenticated = false;
    } finally {
        updateUIForAuthentication();
    }
}

function updateUIForAuthentication() {
    if (state.isAuthenticated && state.currentUser) {
        elements.usernameDisplay.textContent = state.currentUser.username;
        elements.userInitial.textContent = state.currentUser.username[0].toUpperCase();
        elements.loginBtn.style.display = 'none';
        elements.logoutBtn.style.display = 'block';
        loadRecentChats();
    } else {
        elements.usernameDisplay.textContent = 'Guest';
        elements.userInitial.textContent = 'G';
        elements.loginBtn.style.display = 'block';
        elements.logoutBtn.style.display = 'none';
        renderRecentChats([]); // Clear recent chats on logout
    }
}

async function handleModelChange(event) {
    const newModel = event.target.value;
    if (!state.currentAgent || !newModel) return;

    showLoading(true);
    try {
        await apiRequest(`/agents/${state.currentAgent.agent_id}/profile`, {
            method: 'PUT',
            body: { model: newModel }
        });
        showToast(`Agent model updated to ${newModel.split('/').pop()}`);
        state.currentAgent.model = newModel; // Atualiza o estado local
    } catch (error) {
        showError(`Failed to update model: ${error.message}`);
        // Reverter a seleção no dropdown para o valor antigo
        event.target.value = state.currentAgent.model;
    } finally {
        showLoading(false);
    }
}

async function handleAuth(event) {
    event.preventDefault();
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const email = document.getElementById('email').value;
    const isRegister = elements.registerTab.classList.contains('active');

    const endpoint = isRegister ? '/auth/register' : '/auth/login';
    const body = isRegister ? { email, username, password } : { username, password };

    showLoading(true);
    clearError();

    try {
        const data = await apiRequest(endpoint, { method: 'POST', body });
        localStorage.setItem('aura_token', data.access_token);
        await checkAuthentication();
        hideAuthModal();
        showToast(`Welcome ${isRegister ? '' : 'back, '}${state.currentUser.username}!`);
        // If user was trying to access "My Agents", switch to it now
        if (state.currentView === 'my-agents') {
            loadMyAgents();
        }
    } catch (error) {
        showError(error.message || 'Authentication failed.');
    } finally {
        showLoading(false);
    }
}

function logout() {
    localStorage.removeItem('aura_token');
    state.currentUser = null;
    state.isAuthenticated = false;
    state.currentAgent = null;
    state.sessionId = null;
    updateUIForAuthentication();
    switchView('discover');
    showToast('Logged out successfully.');
}

// ==========================================
// VIEW MANAGEMENT
// ==========================================
function switchView(view) {
    state.currentView = view;
    document.querySelectorAll('.view').forEach(v => v.style.display = 'none');
    document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));

    const viewElement = document.getElementById(`${view}-view`);
    const tabElement = document.getElementById(`${view}-tab`);

    if (viewElement) viewElement.style.display = 'block';
    if (tabElement) tabElement.classList.add('active');

    // Load content for the new view
    if (view === 'discover') loadMarketplaceAgents();
    if (view === 'my-agents') {
        if (state.isAuthenticated) loadMyAgents();
        else renderMyAgentsAuthPrompt();
    }
    if (view === 'create') loadCreateView();
}

function toggleSidebar() {
    elements.sidebar.classList.toggle('collapsed');
}

function toggleUserMenu() {
    elements.userMenu.style.display = elements.userMenu.style.display === 'block' ? 'none' : 'block';
}

function toggleAgentDropdown() {
    elements.agentDropdownMenu.style.display = elements.agentDropdownMenu.style.display === 'block' ? 'none' : 'block';
}

function toggleCognitiveDashboard() {
    if (elements.cognitiveDashboard) {
        const isVisible = elements.cognitiveDashboard.style.display === 'block';
        elements.cognitiveDashboard.style.display = isVisible ? 'none' : 'block';
    }
}
// ==========================================
// AGENT & CHAT MANAGEMENT
// ==========================================

async function loadRecentChats() {
    if (!state.isAuthenticated) return;
    try {
        const recentSessions = await apiRequest('/agents'); // The new /agents endpoint lists user agents
        renderRecentChats(recentSessions);
    } catch (error) {
        console.error('Failed to load recent agents/chats:', error);
    }
}

function renderRecentChats(agents) {
    elements.agentsList.innerHTML = '';
    if (!agents || agents.length === 0) {
        elements.agentsList.innerHTML = `<p class="placeholder-text">No agents yet. Create one!</p>`;
        elements.agentCount.textContent = '0';
        return;
    }
    elements.agentCount.textContent = agents.length;
    agents.forEach(agent => {
        const item = document.createElement('div');
        item.className = 'agent-item';
        item.id = `agent-list-item-${agent.agent_id}`;
        item.onclick = () => selectAgent(agent);
        const avatarInitial = agent.name ? agent.name[0].toUpperCase() : 'A';
        item.innerHTML = `
            <div class="agent-avatar"><span>${avatarInitial}</span></div>
            <div class="agent-name">${escapeHtml(agent.name)}</div>
        `;
        elements.agentsList.appendChild(item);
    });
}

async function selectAgent(agent) {
    if (state.currentAgent?.agent_id === agent.agent_id && state.currentView === 'chat') return;

    showLoading(true);
    try {
        // Fetch full details to ensure we have the latest config
        const detailedAgent = await apiRequest(`/agents/${agent.agent_id}`);
        state.currentAgent = detailedAgent;
        state.sessionId = null; // A new session will be created on the first message
        state.messages = [];

        switchView('chat');
        updateChatHeader();
        clearMessages();

        // Load history and then show welcome message if no history exists
        await loadChatHistory(agent.agent_id);
        if (state.messages.length === 0) {
            addMessage('assistant', `Hello! I'm ${agent.name}. How can I help you today?`);
        }

        // Update active selection in sidebar
        document.querySelectorAll('.agent-item').forEach(item => item.classList.remove('active'));
        document.getElementById(`agent-list-item-${agent.agent_id}`)?.classList.add('active');

    } catch (error) {
        showError(`Failed to select agent: ${error.message}`);
        switchView('discover'); // Go back to a safe view
    } finally {
        showLoading(false);
    }
}

async function updateChatHeader() {
    const agent = state.currentAgent;
    if (!agent) return;

    elements.currentAgentName.textContent = agent.name;
    elements.currentAgentDescription.textContent = agent.persona;
    const avatarInitial = agent.name ? agent.name[0].toUpperCase() : 'A';
    elements.currentAgentAvatar.querySelector('span').textContent = avatarInitial;

    // TODO: Populate model selector for chat view if needed in the future
}

async function loadChatHistory(agentId) {
    // The new backend doesn't have a direct history endpoint.
    // Chat history is implicitly handled by session_id on the backend.
    // This function will now just clear the view for a fresh start.
    // The backend will load history when we send the first message with a session_id.
    // For now, we'll simulate a clean slate. A more advanced frontend
    // could store conversations locally in localStorage.
    console.log("Starting a new chat session view for agent:", agentId);
}

async function sendMessage() {
    const message = elements.messageInput.value.trim();
    if (!message || !state.currentAgent || state.isTyping) return;

    addMessage('user', message);
    elements.messageInput.value = '';
    elements.sendBtn.disabled = true;
    showTypingIndicator(true);

    try {
        const payload = {
            message: message,
            session_id: state.sessionId // This will be null on the first message
        };

        const result = await apiRequest(`/agents/${state.currentAgent.agent_id}/chat`, {
            method: 'POST',
            body: payload
        });

        if (result.telemetry) {
            updateCognitiveDashboard(result.telemetry);
        }

        // Adicione a renderização da resposta do assistente
        addMessage('assistant', result.response, result.telemetry); // Passe a telemetria para a função addMessage

        // The backend returns the session_id, save it for subsequent messages
        if (result.session_id) {
            state.sessionId = result.session_id;
        }

        addMessage('assistant', result.response);

    } catch (error) {
        addMessage('system', `Error: ${error.message}`);
    } finally {
        showTypingIndicator(false);
    }
}

// ==========================================
// MARKETPLACE & MY AGENTS
// ==========================================
async function loadMarketplaceAgents() {
    elements.featuredGrid.innerHTML = '<div class="spinner"></div>';
    try {
        // ASSUMPTION: This endpoint exists based on prebuilt_agents_system.py
        const agents = await apiRequest('/prebuilt-agents/list');
        renderMarketplace(agents);
    } catch (error) {
        elements.featuredGrid.innerHTML = `<p class="placeholder-text error-message">Could not load marketplace: ${error.message}</p>`;
    }
}

function renderMarketplace(agents) {
    elements.featuredGrid.innerHTML = '';
    if (!agents || agents.length === 0) {
        elements.featuredGrid.innerHTML = `<p class="placeholder-text">Marketplace is currently empty.</p>`;
        return;
    }

    agents.forEach(agent => {
        const card = document.createElement('div');
        card.className = 'featured-card';
        card.onclick = () => showAgentOptionsModal(agent);

        const avatarInitial = agent.name ? agent.name[0].toUpperCase() : 'A';
        card.innerHTML = `
            <div class="agent-avatar"><span>${avatarInitial}</span></div>
            <h3>${escapeHtml(agent.name)}</h3>
            <p>${escapeHtml(agent.short_description)}</p>
            <div class="agent-meta">
                <span class="system-badge system-badge-${agent.system_type}">${agent.system_type.toUpperCase()}</span>
                <span>${agent.archetype}</span>
            </div>
        `;
        elements.featuredGrid.appendChild(card);
    });
}

async function loadMyAgents() {
    elements.myAgentsGrid.innerHTML = '<div class="spinner"></div>';
    try {
        const agents = await apiRequest('/agents');
        state.agents = agents;
        renderMyAgents(agents);
    } catch (error) {
        renderMyAgentsAuthPrompt(`Could not load agents: ${error.message}`);
    }
}

function renderMyAgents(agents) {
    elements.myAgentsGrid.innerHTML = '';
    if (agents.length === 0) {
        elements.myAgentsGrid.innerHTML = `<p class="placeholder-text">You haven't created or cloned any agents yet.</p>`;
        return;
    }
    agents.forEach(agent => {
        const card = document.createElement('div');
        card.className = 'my-agent-card';
        const avatarInitial = agent.name ? agent.name[0].toUpperCase() : 'A';

        card.innerHTML = `
            <div class="my-agent-card__header">
                <div class="agent-avatar"><span>${avatarInitial}</span></div>
                <div class="my-agent-card__info">
                    <h3>${escapeHtml(agent.name)}</h3>
                    <p>${escapeHtml(agent.persona)}</p>
                </div>
            </div>
            <div class="my-agent-card__details">
                 <div class="detail-item">
                    <span class="label">System</span>
                    <span class="value"><span class="system-badge ceaf">CEAF V3</span></span>
                 </div>
                 <div class="detail-item">
                    <span class="label">Model</span>
                    <span class="value model-value">${escapeHtml(agent.model.split('/').pop())}</span>
                </div>
            </div>
            <div class="my-agent-card__actions">
                <button class="btn-action" onclick="selectAgentById('${agent.agent_id}')">💬 Chat</button>
                <button class="btn-action btn-danger" onclick="deleteAgent('${agent.agent_id}')">🗑️ Delete</button>
            </div>
        `;
        elements.myAgentsGrid.appendChild(card);
    });
}

function renderMyAgentsAuthPrompt(message = "Please sign in to view and manage your agents.") {
    elements.myAgentsGrid.innerHTML = `
        <div class="auth-required-message" style="text-align: center; padding: 40px;">
            <p style="color: var(--text-tertiary); margin-bottom: 24px;">${message}</p>
            <button class="btn-auth" onclick="showAuthModal()">Sign In</button>
        </div>
    `;
}

async function deleteAgent(agentId) {
    if (!confirm('Are you sure you want to permanently delete this agent and all its data?')) return;
    showLoading(true);
    try {
        await apiRequest(`/agents/${agentId}`, { method: 'DELETE' });
        showToast('Agent deleted successfully.');
        // Refresh both "My Agents" and the sidebar list
        await loadMyAgents();
        await loadRecentChats();
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

// ==========================================
// AGENT CREATION
// ==========================================
async function loadCreateView() {
    await populateModelSelector(elements.modelSelectCreate);
    validateCreateForm();
}

async function populateModelSelector(selectElement) {
    if (!selectElement) return;
    selectElement.innerHTML = '<option value="">Loading models...</option>';
    try {
        const modelsData = await apiRequest('/models/openrouter'); // Endpoint CORRIGIDO
        selectElement.innerHTML = '<option value="">Select a model...</option>';

        // Lógica CORRIGIDA para processar a resposta em categorias
        for (const category in modelsData) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = category;
            modelsData[category].forEach(model => {
                const option = document.createElement('option');
                option.value = model.name;
                option.textContent = `${model.name.split('/').pop()} (${model.cost_display})`;
                optgroup.appendChild(option);
            });
            selectElement.appendChild(optgroup);
        }
    } catch (error) {
        selectElement.innerHTML = '<option value="">Error loading models</option>';
        console.error("Failed to populate models:", error);
    }
}

function validateCreateForm() {
    const name = document.getElementById('agent-name').value.trim();
    const persona = document.getElementById('agent-persona').value.trim();
    const detailed = document.getElementById('agent-detailed-persona').value.trim();
    const model = elements.modelSelectCreate.value;
    elements.createButton.disabled = !(name && persona && detailed && model);
}

async function handleAgentCreation(event) {
    event.preventDefault();
    if (!state.isAuthenticated) {
        showError('Please log in to create an agent.');
        return;
    }

    const agentData = {
        name: document.getElementById('agent-name').value,
        persona: document.getElementById('agent-persona').value,
        detailed_persona: document.getElementById('agent-detailed-persona').value,
        model: elements.modelSelectCreate.value,
        settings: {
            system_type: "ceaf_v3" // All agents are now V3
        }
    };

    showLoading(true);
    try {
        const newAgent = await apiRequest('/agents', { method: 'POST', body: agentData });
        showToast(`Agent "${agentData.name}" created successfully!`);
        elements.agentForm.reset();
        // Refresh user's agent lists and switch to the new agent
        await loadRecentChats();
        await selectAgent({ agent_id: newAgent.agent_id, name: agentData.name }); // Switch to the new agent
    } catch (error) {
        showError(`Creation failed: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

// ==========================================
// RAG - KNOWLEDGE FILES MODAL
// ==========================================
async function showFilesModal() {
    if (!state.currentAgent) return;
    elements.filesModal.style.display = 'flex';
    document.body.style.overflow = 'hidden';
    toggleAgentDropdown();
    // In CEAF V3, there's no endpoint to list files, so we show a static message.
    elements.filesListContainer.innerHTML = `<p class="placeholder-text">Uploaded files are processed and stored internally. You can upload new files below.</p>`;
}

function closeFilesModal() {
    elements.filesModal.style.display = 'none';
    document.body.style.overflow = 'auto';
}

async function handleKnowledgeFileUpload({ target }) {
    const file = target.files[0];
    if (!file || !state.currentAgent) return;

    showLoading(true);
    try {
        const formData = new FormData();
        formData.append('file', file);

        const result = await apiRequest(`/agents/${state.currentAgent.agent_id}/files/upload`, {
            method: 'POST',
            body: formData, // FormData is sent directly, not as JSON
            isJson: false,
        });

        showToast(result.message || "File uploaded and indexed successfully.");
        // No need to refresh a list, just give feedback.
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
        target.value = ''; // Reset file input
    }
}

// ==========================================
// CHAT UI & MESSAGING
// ==========================================
function addMessage(role, content, telemetry = null) { // Adiciona parâmetro opcional
    const message = { role, content, timestamp: new Date(), telemetry }; // Armazena a telemetria com a mensagem
    state.messages.push(message);
    renderMessage(message);
    scrollToBottom();
}

function renderMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${message.role}`;

    // Lida com mensagens de sistema (erros, etc.) que não têm avatar
    if (message.role === 'system') {
        messageDiv.innerHTML = `<div class="message-content">${escapeHtml(message.content)}</div>`;
        elements.chatMessages.appendChild(messageDiv);
        return; // Encerra a função aqui para mensagens de sistema
    }

    // Processamento para mensagens de 'user' e 'assistant'

    // 1. Prepara o conteúdo da mensagem
    let messageContent = escapeHtml(message.content);

    // 2. FEATURE: Citações de Memória Interativas
    // Procura por (mem_xxxx...) e transforma em links clicáveis
    // Esta regex é mais segura e captura o ID corretamente.
    messageContent = messageContent.replace(/\(mem_([a-fA-F0-9]{4,})\)/g, (match, memId) => {
        // O `match` é o texto completo, ex: "(mem_743a)"
        // O `memId` é apenas o ID capturado, ex: "743a"
        // Criamos um ID de memória completo para a função. Aqui assumimos que os 4 primeiros caracteres são suficientes para o link, mas passamos o ID completo se possível.
        // A melhor abordagem é passar o match completo.
        return `<a href="#" class="memory-link" onclick="showMemory('${match}')" title="Click to see memory details">${match}</a>`;
    });


    // 3. FEATURE: Renderização de Markdown para tabelas, listas, etc.
    // Garante que a biblioteca marked.js foi incluída no HTML
    if (typeof marked !== 'undefined') {
        messageContent = marked.parse(messageContent);
    } else {
        // Fallback para quebras de linha simples se marked.js não estiver disponível
        messageContent = messageContent.replace(/\n/g, '<br>');
    }

    // 4. FEATURE: Indicador de Confiança
    let confidenceHTML = '';
    if (message.role === 'assistant' && message.telemetry && message.telemetry.turn_final_confidence) {
        const confidence = message.telemetry.turn_final_confidence * 100;
        confidenceHTML = `
            <div class="confidence-indicator" title="Confidence: ${confidence.toFixed(0)}%">
                <div class="confidence-bar" style="width: ${confidence}%;"></div>
            </div>
        `;
    }

    // 5. Determina o avatar a ser usado
    const avatarInitial = message.role === 'user'
        ? (state.currentUser?.username?.[0].toUpperCase() || 'U')
        : (state.currentAgent?.name?.[0].toUpperCase() || 'A');

    // 6. Monta o HTML final da mensagem
    messageDiv.innerHTML = `
        <div class="agent-avatar"><span>${avatarInitial}</span></div>
        <div class="message-content-wrapper">
             <div class="message-content">${messageContent}</div>
             ${confidenceHTML}
        </div>
    `;

    // 7. Adiciona a mensagem final ao chat
    elements.chatMessages.appendChild(messageDiv);
}

function clearMessages() {
    elements.chatMessages.innerHTML = '';
    state.messages = [];
}

function showTypingIndicator(show) {
    state.isTyping = show;
    elements.sendBtn.disabled = show || elements.messageInput.value.trim().length === 0;

    const existingIndicator = document.getElementById('typing-indicator');
    if (existingIndicator) existingIndicator.remove();

    if (show) {
        const typingDiv = document.createElement('div');
        typingDiv.id = 'typing-indicator';
        typingDiv.className = 'message assistant';
        const avatar = state.currentAgent.name[0].toUpperCase();
        typingDiv.innerHTML = `
            <div class="agent-avatar"><span>${avatar}</span></div>
            <div class="message-content"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div>
        `;
        elements.chatMessages.appendChild(typingDiv);
        scrollToBottom();
    }
}

function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// ==========================================
// MODAL & UI HELPERS
// ==========================================
function showAuthModal() {
    elements.authModal.style.display = 'flex';
}

function hideAuthModal() {
    elements.authModal.style.display = 'none';
}

function switchAuthTab(mode) {
    elements.loginTab.classList.toggle('active', mode === 'login');
    elements.registerTab.classList.toggle('active', mode === 'register');
    elements.emailGroup.style.display = mode === 'register' ? 'block' : 'none';
    elements.authSubmitBtn.textContent = mode === 'login' ? 'Sign in' : 'Sign up';
    document.getElementById('email').required = mode === 'register';
}

function showAgentOptionsModal(agent) {
    // Clean up any old modal
    const oldOverlay = document.getElementById('agent-options-overlay');
    if (oldOverlay) oldOverlay.remove();

    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.id = 'agent-options-overlay';
    overlay.style.display = 'flex';
    overlay.onclick = (e) => { if (e.target === overlay) e.currentTarget.remove(); };

    const modal = document.createElement('div');
    modal.className = 'modal';
    const avatarInitial = agent.name ? agent.name[0].toUpperCase() : 'A';

    modal.innerHTML = `
        <div class="modal-header">
            <h2>${escapeHtml(agent.name)}</h2>
            <button class="modal-close" onclick="document.getElementById('agent-options-overlay').remove()">&times;</button>
        </div>
        <div class="modal-content">
            <p style="color: var(--text-secondary); margin-bottom: 24px;">${escapeHtml(agent.short_description)}</p>
            <div class="modal-actions">
                <button class="btn-auth" onclick="cloneAndChat('${agent.id}')">Clone & Chat</button>
            </div>
        </div>
    `;
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
}

async function showMemory(memId) {
    // Remove parênteses e o prefixo 'mem_'
    const cleanId = memId.replace(/[()]/g, '');
    showLoading(true);
    try {
        // **BACKEND TASK:** Você precisará criar um novo endpoint na API
        // GET /agents/{agent_id}/memories/{memory_id} para buscar uma memória específica.
        const memoryData = await apiRequest(`/agents/${state.currentAgent.agent_id}/memories/${cleanId}`);

        // Exibir a memória em um modal simples
        showToast(`Memory [${cleanId}]: ${memoryData.content}`, 'info');

    } catch (error) {
        showError(`Could not retrieve memory: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

async function cloneAndChat(agentId) {
    if (!state.isAuthenticated) {
        showAuthModal();
        return;
    }
    showLoading(true);
    try {
        // Payload correto para o endpoint /agents/clone
        const payload = {
            source_agent_id: agentId,
            clone_memories: true
        };
        const result = await apiRequest(`/agents/clone`, { method: 'POST', body: payload }); // Endpoint CORRIGIDO

        showToast(`Agent "${result.name}" cloned successfully!`);
        await loadRecentChats(); // Refresh sidebar

        // O resultado da clonagem já dá o ID do novo agente
        await selectAgent({ agent_id: result.agent_id, name: result.name });

    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
        const overlay = document.getElementById('agent-options-overlay');
        if (overlay) overlay.remove();
    }
}

// Helper to find agent by ID for UI updates
function selectAgentById(agentId) {
    const agent = state.agents.find(a => a.agent_id === agentId);
    if (agent) {
        selectAgent(agent);
    } else {
        showError("Could not find agent. Please refresh.");
    }
}

// ==========================================
// UTILITY FUNCTIONS
// ==========================================
async function apiRequest(endpoint, options = {}) {
    const { method = 'GET', body = null, isJson = true } = options;
    const headers = { ...options.headers };

    if (state.isAuthenticated) {
        headers['Authorization'] = `Bearer ${localStorage.getItem('aura_token')}`;
    }

    if (body && isJson) {
        headers['Content-Type'] = 'application/json';
    }

    const config = {
        method,
        headers,
        body: body ? (isJson ? JSON.stringify(body) : body) : null,
    };

    const response = await fetch(`${CONFIG.API_BASE}${endpoint}`, config);

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'An unknown error occurred.' }));
        throw new Error(errorData.detail || `HTTP Error ${response.status}`);
    }
    
    // Handle responses with no content (e.g., DELETE 204)
    if (response.status === 204) {
        return null;
    }

    return response.json();
}

function escapeHtml(text) {
    if (typeof text !== 'string') return '';
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function showLoading(show) {
    elements.loadingOverlay.style.display = show ? 'flex' : 'none';
}

function showError(message) {
    if (elements.errorContainer) {
        elements.errorContainer.textContent = message;
        elements.errorContainer.style.display = 'block';
    }
    showToast(message, 'error');
}

function clearError() {
    if (elements.errorContainer) {
        elements.errorContainer.style.display = 'none';
        elements.errorContainer.textContent = '';
    }
}

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 500);
        }, 3000);
    }, 10);
}

// Make functions globally available for inline onclick handlers
window.switchView = switchView;
window.showAuthModal = showAuthModal;
window.hideAuthModal = hideAuthModal;
window.switchAuthTab = switchAuthTab;
window.handleAuth = handleAuth;
window.logout = logout;
window.selectAgentById = selectAgentById;
window.deleteAgent = deleteAgent;
window.showFilesModal = showFilesModal;
window.closeFilesModal = closeFilesModal;
window.showAgentOptionsModal = showAgentOptionsModal;
window.cloneAndChat = cloneAndChat;
window.showMemory = showMemory;