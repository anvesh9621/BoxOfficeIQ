-- ============================================================
-- BoxOfficeIQ Seed Data: 15 Theatres
-- Matches synthetic dataset generated for BoxOfficeIQ
-- ============================================================

INSERT INTO theatres (theatre_id, name, theatre_type, city, state, capacity, is_active)
VALUES
    (1,  'PVR Cinemas Juhu',         'Multiplex',    'Mumbai',    'Maharashtra', 350, TRUE),
    (2,  'INOX Nariman Point',       'Multiplex',    'Mumbai',    'Maharashtra', 320, TRUE),
    (3,  'Cinepolis Andheri',        'Multiplex',    'Mumbai',    'Maharashtra', 290, TRUE),
    (4,  'Regal Cinema Colaba',      'SingleScreen', 'Mumbai',    'Maharashtra', 600, TRUE),
    (5,  'IMAX Wadala',              'IMAX',         'Mumbai',    'Maharashtra', 500, TRUE),
    (6,  'PVR Select Citywalk',      'Multiplex',    'Delhi',     'Delhi',       310, TRUE),
    (7,  'INOX Nehru Place',         'Multiplex',    'Delhi',     'Delhi',       280, TRUE),
    (8,  'Delite Cinema Daryaganj',  'SingleScreen', 'Delhi',     'Delhi',       550, TRUE),
    (9,  'PVR Forum Bengaluru',      'Multiplex',    'Bengaluru', 'Karnataka',   300, TRUE),
    (10, 'Cinepolis Orion Mall',     'Multiplex',    'Bengaluru', 'Karnataka',   270, TRUE),
    (11, 'IMAX UB City',             'IMAX',         'Bengaluru', 'Karnataka',   480, TRUE),
    (12, 'Sathyam Cinemas',          'Multiplex',    'Chennai',   'Tamil Nadu',  340, TRUE),
    (13, 'Rohini Silver Screens',    'SingleScreen', 'Chennai',   'Tamil Nadu',  580, TRUE),
    (14, 'Prasads Multiplex IMAX',   'IMAX',         'Hyderabad', 'Telangana',   510, TRUE),
    (15, 'Star Drive-In Ahmedabad',  'DriveIn',      'Ahmedabad', 'Gujarat',     200, TRUE)
ON CONFLICT (theatre_id) DO UPDATE SET
    name = EXCLUDED.name,
    theatre_type = EXCLUDED.theatre_type,
    city = EXCLUDED.city,
    state = EXCLUDED.state,
    capacity = EXCLUDED.capacity,
    is_active = EXCLUDED.is_active;
