INSERT INTO global_property (property, property_value, description, uuid) VALUES ('authentication.scheme', 'basic', 'Enable basic auth', UUID()) ON DUPLICATE KEY UPDATE property_value='basic';
