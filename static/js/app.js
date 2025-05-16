document.addEventListener('DOMContentLoaded', function() {
    // Referencias a elementos del DOM
    const btnStart = document.getElementById('btnStart');
    const btnCollect = document.getElementById('btnCollect');
    const btnTrain = document.getElementById('btnTrain');
    const btnStop = document.getElementById('btnStop');
    const status = document.getElementById('status');
    const detectedSign = document.getElementById('detectedSign');
    const probabilities = document.getElementById('probabilities');
    
    // Variable para controlar la actualización de resultados
    let resultInterval = null;
    
    // Función para obtener resultados periódicamente
    function fetchResults() {
        fetch('/get_results')
            .then(response => response.json())
            .then(data => {
                if (data.current_action) {
                    detectedSign.textContent = data.current_action;
                    updateProbabilitiesFromData(data.probabilities);
                }
            })
            .catch(error => {
                console.error('Error al obtener resultados:', error);
            });
    }
    
    // Actualizar las probabilidades con datos del servidor
    function updateProbabilitiesFromData(probData) {
        // Limpiar contenedor de probabilidades
        probabilities.innerHTML = "";
        
        // Validar que probData sea un objeto
        if (!probData || typeof probData !== 'object') {
            console.error('Datos de probabilidad inválidos:', probData);
            return;
        }
        
        // Agregar cada probabilidad
        Object.entries(probData).forEach(([action, prob]) => {
            const item = document.createElement('div');
            item.className = 'probability-item';
            
            // Asegurar que la probabilidad sea un número
            const probability = typeof prob === 'number' ? prob : 0;
            
            item.innerHTML = `
                <span>${action}</span>
                <span>${probability.toFixed(2)}</span>
                <div class="progress-bar">
                    <div class="progress" style="width: ${probability * 100}%"></div>
                </div>
            `;
            probabilities.appendChild(item);
        });
    }
    
    // Función para limpiar las probabilidades
    function clearProbabilities() {
        probabilities.innerHTML = "";
        
        // Obtener acciones del servidor (definido en la plantilla HTML)
        const actions = window.availableActions || [];
        
        // Si no hay acciones disponibles, usar algunas predeterminadas
        const displayActions = actions.length > 0 ? 
            actions.slice(0, 4) : 
            ['hola', 'gracias', 'por favor', 'ayuda'];
        
        // Mostrar acciones con probabilidad cero
        displayActions.forEach(action => {
            const item = document.createElement('div');
            item.className = 'probability-item';
            item.innerHTML = `
                <span>${action}</span>
                <span>0.00</span>
                <div class="progress-bar">
                    <div class="progress" style="width: 0%"></div>
                </div>
            `;
            probabilities.appendChild(item);
        });
    }
    
    // Funciones de los botones
    if (btnStart) {
        btnStart.addEventListener('click', function() {
            // Mostrar spinner o indicador de carga
            status.textContent = "Iniciando traductor...";
            status.style.backgroundColor = "#d5f5e3";
            
            fetch('/start_detection')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Error en la respuesta del servidor');
                    }
                    return response.json();
                })
                .then(data => {
                    status.textContent = data.status || "Traductor iniciado";
                    
                    // Detener intervalo existente si hay uno
                    if (resultInterval) {
                        clearInterval(resultInterval);
                    }
                    
                    // Iniciar actualización periódica de resultados
                    resultInterval = setInterval(fetchResults, 500);
                })
                .catch(error => {
                    console.error('Error:', error);
                    status.textContent = "Error al iniciar el traductor";
                    status.style.backgroundColor = "#fadbd8";
                });
        });
    }
    
    if (btnCollect) {
        btnCollect.addEventListener('click', function() {
            status.textContent = "Iniciando recopilación de datos...";
            status.style.backgroundColor = "#d4efdf";
            
            fetch('/collect_data')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Error en la respuesta del servidor');
                    }
                    return response.json();
                })
                .then(data => {
                    status.textContent = data.status || "Recopilación iniciada";
                })
                .catch(error => {
                    console.error('Error:', error);
                    status.textContent = "Error en la recopilación de datos";
                    status.style.backgroundColor = "#fadbd8";
                });
        });
    }
    
    if (btnTrain) {
        btnTrain.addEventListener('click', function() {
            status.textContent = "Iniciando entrenamiento...";
            status.style.backgroundColor = "#fef9e7";
            
            fetch('/train_model')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Error en la respuesta del servidor');
                    }
                    return response.json();
                })
                .then(data => {
                    status.textContent = data.status || "Entrenamiento iniciado";
                })
                .catch(error => {
                    console.error('Error:', error);
                    status.textContent = "Error en el entrenamiento";
                    status.style.backgroundColor = "#fadbd8";
                });
        });
    }
    
    if (btnStop) {
        btnStop.addEventListener('click', function() {
            status.textContent = "Deteniendo...";
            status.style.backgroundColor = "#fadbd8";
            
            fetch('/stop_detection')
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Error en la respuesta del servidor');
                    }
                    return response.json();
                })
                .then(data => {
                    status.textContent = data.status || "Detección detenida";
                    detectedSign.textContent = "Esperando señas...";
                    
                    // Detener intervalo de actualización
                    if (resultInterval) {
                        clearInterval(resultInterval);
                        resultInterval = null;
                    }
                    
                    clearProbabilities();
                })
                .catch(error => {
                    console.error('Error:', error);
                    status.textContent = "Error al detener";
                });
        });
    }
    
    // Inicializar la vista
    clearProbabilities();
});