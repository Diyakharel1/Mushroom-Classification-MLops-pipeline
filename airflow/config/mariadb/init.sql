-- Initialize MariaDB database for mushroom analytics

-- Create database if not exists (should already exist from environment)
CREATE DATABASE IF NOT EXISTS mushroom_analytics;
USE mushroom_analytics;

-- Grant privileges to the mushroom_user
GRANT ALL PRIVILEGES ON mushroom_analytics.* TO 'mushroom_user'@'%';
FLUSH PRIVILEGES;

-- Create tables with InnoDB engine (compatible with all MariaDB versions)
CREATE TABLE IF NOT EXISTS cleaned_features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    cap_diameter DECIMAL(10,6),
    cap_shape VARCHAR(50),
    cap_surface VARCHAR(50),
    cap_color VARCHAR(50),
    does_bruise_or_bleed VARCHAR(50),
    gill_attachment VARCHAR(50),
    gill_spacing VARCHAR(50),
    gill_color VARCHAR(50),
    stem_height DECIMAL(10,6),
    stem_width DECIMAL(10,6),
    stem_root VARCHAR(50),
    stem_surface VARCHAR(50),
    stem_color VARCHAR(50),
    veil_type VARCHAR(50),
    veil_color VARCHAR(50),
    has_ring VARCHAR(50),
    ring_type VARCHAR(50),
    spore_print_color VARCHAR(50),
    habitat VARCHAR(50),
    season VARCHAR(50),
    class VARCHAR(10),
    data_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_class (class),
    INDEX idx_data_version (data_version)
) ENGINE=InnoDB;

-- Create data split tables
CREATE TABLE IF NOT EXISTS train_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    feature_id INT,
    experiment_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_experiment_id (experiment_id),
    INDEX idx_feature_id (feature_id),
    FOREIGN KEY (feature_id) REFERENCES cleaned_features(id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS test_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    feature_id INT,
    experiment_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_experiment_id (experiment_id),
    INDEX idx_feature_id (feature_id),
    FOREIGN KEY (feature_id) REFERENCES cleaned_features(id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS validation_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    feature_id INT,
    experiment_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_experiment_id (experiment_id),
    INDEX idx_feature_id (feature_id),
    FOREIGN KEY (feature_id) REFERENCES cleaned_features(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Create experiment metadata table
CREATE TABLE IF NOT EXISTS experiments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id VARCHAR(100) UNIQUE,
    name VARCHAR(200),
    description TEXT,
    status VARCHAR(50) DEFAULT 'created',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_experiment_id (experiment_id),
    INDEX idx_status (status)
) ENGINE=InnoDB;

-- Create data lineage table
CREATE TABLE IF NOT EXISTS data_lineage (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id VARCHAR(100),
    source_file VARCHAR(500),
    processing_step VARCHAR(100),
    input_records INT,
    output_records INT,
    processing_time DECIMAL(10,4),
    status VARCHAR(50),
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_experiment_id (experiment_id),
    INDEX idx_processing_step (processing_step)
) ENGINE=InnoDB;

-- Insert sample experiment for testing
INSERT INTO experiments (experiment_id, name, description, status) 
VALUES ('test_exp_001', 'Initial Test Experiment', 'Testing database connectivity and basic functionality', 'created')
ON DUPLICATE KEY UPDATE 
updated_at = CURRENT_TIMESTAMP;

-- Show tables to verify creation
SHOW TABLES;

-- Show table structures
DESCRIBE cleaned_features;
DESCRIBE experiments;

-- Show engine status
SHOW ENGINES;

-- Test data insertion
INSERT INTO cleaned_features (
    cap_diameter, cap_shape, cap_surface, cap_color, 
    does_bruise_or_bleed, gill_attachment, gill_spacing, gill_color,
    stem_height, stem_width, stem_root, stem_surface, stem_color,
    veil_type, veil_color, has_ring, ring_type, spore_print_color,
    habitat, season, class, data_version
) VALUES (
    5.5, 'convex', 'smooth', 'brown',
    'no', 'free', 'close', 'white',
    8.2, 1.3, 'bulbous', 'smooth', 'white',
    'partial', 'white', 'yes', 'pendant', 'white',
    'grasses', 'autumn', 'edible', 'test_v1'
);

-- Verify test insertion
SELECT COUNT(*) as test_count FROM cleaned_features;
