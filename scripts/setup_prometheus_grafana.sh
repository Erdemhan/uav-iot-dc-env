#!/bin/bash
# setup_prometheus_grafana.sh
# Installs and configures Prometheus and Grafana for Ray Dashboard on WSL2 Ubuntu (Head Node).

set -e

echo "=================================================="
echo " Installing Prometheus and Grafana for Ray Cluster"
echo "=================================================="

# 1. Install Prometheus
echo "Installing Prometheus..."
sudo apt-get update
sudo apt-get install -y prometheus

# 2. Install Grafana
echo "Installing Grafana..."
sudo apt-get install -y apt-transport-https software-properties-common wget
sudo mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt-get update
sudo apt-get install -y grafana

# 3. Configure Prometheus for Ray Service Discovery
echo "Configuring Prometheus..."
PROMETHEUS_CONF="/etc/prometheus/prometheus.yml"

# Backup original config
if [ ! -f "${PROMETHEUS_CONF}.bak" ]; then
    sudo cp "$PROMETHEUS_CONF" "${PROMETHEUS_CONF}.bak"
fi

# Create a clean Prometheus config with Ray scraping
sudo tee "$PROMETHEUS_CONF" > /dev/null << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'ray'
    file_sd_configs:
      - files:
          - '/tmp/ray/prom_metrics_service_discovery.json'
EOF

# 4. Provision Grafana Datasource (Prometheus)
echo "Configuring Grafana Datasource..."
sudo mkdir -p /etc/grafana/provisioning/datasources
sudo tee /etc/grafana/provisioning/datasources/prometheus.yaml > /dev/null << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
    editable: true
EOF

# 5. Provision Grafana Dashboard (Ray Dashboard)
echo "Configuring Grafana Dashboards..."
sudo mkdir -p /etc/grafana/provisioning/dashboards
sudo tee /etc/grafana/provisioning/dashboards/ray.yaml > /dev/null << 'EOF'
apiVersion: 1

providers:
  - name: 'Ray'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
EOF

# Create dashboard folder and copy Ray's built-in dashboards
sudo mkdir -p /var/lib/grafana/dashboards
echo "Copying Ray's built-in Grafana Dashboards..."
if [ -d "/tmp/ray/session_latest/metrics/grafana/dashboards" ]; then
    sudo cp /tmp/ray/session_latest/metrics/grafana/dashboards/*.json /var/lib/grafana/dashboards/
    echo "Successfully copied all dashboards from /tmp/ray/session_latest/metrics/grafana/dashboards/"
else
    echo "Warning: /tmp/ray/session_latest/metrics/grafana/dashboards/ not found. Start Ray cluster first to generate them."
fi

# Fix permissions for Grafana dashboards folder
sudo chown -R grafana:grafana /var/lib/grafana/dashboards

# 5.5 Configure allow_embedding and anonymous auth in /etc/grafana/grafana.ini
echo "Configuring security and anonymous auth in /etc/grafana/grafana.ini..."
sudo python3 -c "
import re
try:
    with open('/etc/grafana/grafana.ini', 'r') as f:
        content = f.read()

    # Replace allow_embedding under [security]
    security_sec = re.search(r'\[security\](.*?)(?=\n\[|$)', content, re.DOTALL)
    if security_sec:
        sec_content = security_sec.group(1)
        if 'allow_embedding' in sec_content:
            new_sec_content = re.sub(r';?\s*allow_embedding\s*=\s*\w+', 'allow_embedding = true', sec_content)
            content = content.replace(sec_content, new_sec_content)
        else:
            new_sec_content = sec_content + '\nallow_embedding = true\n'
            content = content.replace(sec_content, new_sec_content)

    # Replace enabled and org_role under [auth.anonymous]
    auth_sec = re.search(r'\[auth\.anonymous\](.*?)(?=\n\[|$)', content, re.DOTALL)
    if auth_sec:
        sec_content = auth_sec.group(1)
        new_sec_content = sec_content
        if 'enabled' in sec_content:
            new_sec_content = re.sub(r';?\s*enabled\s*=\s*\w+', 'enabled = true', new_sec_content)
        else:
            new_sec_content += '\nenabled = true\n'
        if 'org_role' in sec_content:
            new_sec_content = re.sub(r';?\s*org_role\s*=\s*\w+', 'org_role = Viewer', new_sec_content)
        else:
            new_sec_content += '\norg_role = Viewer\n'
        content = content.replace(sec_content, new_sec_content)

    with open('/etc/grafana/grafana.ini', 'w') as f:
        f.write(content)
    print('[OK] Grafana security and anonymous auth configured successfully.')
except Exception as e:
    print(f'[WARN] Failed to configure grafana.ini automatically: {e}')
"

# 6. Restart Services
echo "Restarting services..."
restart_service() {
    service_name=$1
    if systemctl is-active --quiet dbus 2>/dev/null || systemctl is-system-running --quiet 2>/dev/null; then
        echo "Using systemctl to restart $service_name..."
        sudo systemctl restart "$service_name" || true
        sudo systemctl enable "$service_name" || true
    else
        echo "Using service command to restart $service_name..."
        sudo service "$service_name" restart || true
    fi
}

restart_service prometheus
restart_service grafana-server

echo "=================================================="
echo " [SUCCESS] Prometheus and Grafana are configured!"
echo " - Prometheus is running on: http://localhost:9090"
echo " - Grafana is running on: http://localhost:3000"
echo " - Ray Dashboard should now display time-series charts."
echo "=================================================="
