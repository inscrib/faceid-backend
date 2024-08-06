use candid::{CandidType, Deserialize, Principal};
use ic_cdk::api;
use ic_cdk::caller;
use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager},
    DefaultMemoryImpl,
};
use onnx::{setup, BoundingBox, Embedding, Person};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

mod benchmarking;
mod onnx;
mod storage;

const WASI_MEMORY_ID: MemoryId = MemoryId::new(0);

const FACE_DETECTION_FILE: &str = "face-detection.onnx";
const FACE_RECOGNITION_FILE: &str = "face-recognition.onnx";

thread_local! {
    static RECOGNITION_ATTEMPTS: RefCell<HashMap<Principal, u32>> = RefCell::new(HashMap::new());
    static RECOGNITION_RESULTS: RefCell<HashMap<Principal, (String, f32)>> = RefCell::new(HashMap::new());

    static ADD_CALLERS: RefCell<HashSet<Principal>> = RefCell::new(HashSet::new());
    static ADD_COUNT: RefCell<usize> = RefCell::new(0);
    static IS_ENABLED: RefCell<bool> = RefCell::new(false);
    // The memory manager is used for simulating multiple memories.
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
}

const MAX_ADD_CALLS: usize = 200;

const ADMIN_PRINCIPAL: &str = "4s4hz-og66m-hypzp-uxv6q-addgn-hshem-dnvln-uhy7t-h3hsc-pmajb-mqe";

const MAX_ATTEMPTS: u32 = 3;

#[derive(CandidType, Deserialize)]
struct RecognitionResult {
    label: String,
    score: f32,
}

#[derive(CandidType, Deserialize)]
struct Error {
    message: String,
}

#[derive(CandidType, Deserialize)]
enum Detection {
    Ok(BoundingBox),
    Err(Error),
}

#[derive(CandidType, Deserialize)]
enum Addition {
    Ok(Embedding),
    Err(Error),
}

#[derive(CandidType, Deserialize)]
enum Recognition {
    Ok(Person),
    Err(Error),
}

fn detect(image: Vec<u8>) -> Detection {
    let result: Detection = match onnx::detect(image) {
        Ok(result) => Detection::Ok(result.0),
        Err(err) => Detection::Err(Error {
            message: err.to_string(),
        }),
    };
    result
}

#[ic_cdk::update]
fn recognize(image: Vec<u8>) -> Recognition {
    let caller = ic_cdk::caller();

    if !ADD_CALLERS.with(|callers| callers.borrow().contains(&caller)) {
        return Recognition::Err(Error {
            message: "Unauthorized: User not in the allowed set".to_string(),
        });
    }

    if RECOGNITION_RESULTS.with(|results| results.borrow().contains_key(&caller)) {
        return Recognition::Err(Error {
            message: "Recognition already successful. Further attempts not allowed".to_string(),
        });
    }

    let attempts = RECOGNITION_ATTEMPTS.with(|attempts| {
        let mut attempts = attempts.borrow_mut();
        let count = attempts.entry(caller).or_insert(0);
        *count += 1;
        *count
    });

    if attempts > MAX_ATTEMPTS {
        return Recognition::Err(Error {
            message: "Maximum recognition attempts exceeded".to_string(),
        });
    }

    match onnx::recognize(image) {
        Ok(person) => {
            RECOGNITION_RESULTS.with(|results| {
                results
                    .borrow_mut()
                    .insert(caller, (person.label.clone(), person.score));
            });

            Recognition::Ok(person)
        }
        Err(e) => {
            if attempts == MAX_ATTEMPTS {
                Recognition::Err(Error {
                    message: format!("Recognition failed after {} attempts", MAX_ATTEMPTS),
                })
            } else {
                Recognition::Err(Error {
                    message: e.to_string(),
                })
            }
        }
    }
}

/// Adds a person with the given name (label) and face (image) for future
/// face recognition requests.
// #[ic_cdk::update]
// fn add(label: String, image: Vec<u8>, code: String) -> Addition {
//     let caller = caller();

//     // Check if the caller is anonymous
//     if caller == Principal::anonymous() {
//         return Addition::Err(Error {
//             message: "Anonymous callers are not allowed".to_string(),
//         });
//     }

//     if code != "qMu11Dfmw" {
//         return Addition::Err(Error {
//             message: "Unauthorized frontend access".to_string(),
//         });
//     }

//     // Check if the function is enabled
//     if !IS_ENABLED.with(|enabled| *enabled.borrow()) {
//         return Addition::Err(Error {
//             message: "This function is currently disabled".to_string(),
//         });
//     }

//     // Check if the caller has already added a face
//     if ADD_CALLERS.with(|callers| callers.borrow().contains(&caller)) {
//         return Addition::Err(Error {
//             message: "You have already added a face".to_string(),
//         });
//     }

//     // Check if the maximum number of calls has been reached
//     if ADD_COUNT.with(|count| *count.borrow() >= MAX_ADD_CALLS) {
//         return Addition::Err(Error {
//             message: "Maximum number of add calls reached".to_string(),
//         });
//     }

//     // Perform the add operation
//     let result = match onnx::add(label, image) {
//         Ok(result) => {
//             // Update the state
//             ADD_CALLERS.with(|callers| callers.borrow_mut().insert(caller));
//             ADD_COUNT.with(|count| *count.borrow_mut() += 1);
//             Addition::Ok(result)
//         },
//         Err(err) => Addition::Err(Error {
//             message: err.to_string(),
//         }),
//     };

//     result
// }

#[ic_cdk::update]
fn add(label: String, image: Vec<u8>, code: String) -> Addition {
    let caller = caller();

    if caller == Principal::anonymous() {
        return Addition::Err(Error {
            message: "Anonymous callers are not allowed".to_string(),
        });
    }

    if code != "qMu11Dfmw" {
        return Addition::Err(Error {
            message: "Unauthorized frontend access".to_string(),
        });
    }

    // Check if the function is enabled
    if !IS_ENABLED.with(|enabled| *enabled.borrow()) {
        return Addition::Err(Error {
            message: "This function is currently disabled".to_string(),
        });
    }

    if ADD_CALLERS.with(|callers| callers.borrow().contains(&caller)) {
        return Addition::Err(Error {
            message: "You have already added a face".to_string(),
        });
    }

    if ADD_COUNT.with(|count| *count.borrow() >= MAX_ADD_CALLS) {
        return Addition::Err(Error {
            message: "Maximum number of add calls reached".to_string(),
        });
    }

    if RECOGNITION_RESULTS.with(|results| {
        results
            .borrow()
            .values()
            .any(|(rec_label, _)| rec_label == &label)
    }) {
        return Addition::Err(Error {
            message: "This face has already been recognized and cannot be added again".to_string(),
        });
    }

    let result = match onnx::add(label, image) {
        Ok(result) => {
            ADD_CALLERS.with(|callers| callers.borrow_mut().insert(caller));
            ADD_COUNT.with(|count| *count.borrow_mut() += 1);
            Addition::Ok(result)
        }
        Err(err) => Addition::Err(Error {
            message: err.to_string(),
        }),
    };

    result
}

#[ic_cdk::update]
fn toggle_add_function(enable: bool) -> Result<(), String> {
    let caller = ic_cdk::caller();
    let admin =
        Principal::from_text(ADMIN_PRINCIPAL).map_err(|_| "Invalid admin principal".to_string())?;

    if caller != admin {
        return Err("Only the admin can toggle this function".to_string());
    }

    IS_ENABLED.with(|enabled| *enabled.borrow_mut() = enable);
    Ok(())
}

#[ic_cdk::update]
fn clear_face_detection_model_bytes() -> CanisterResponse<()> {
    match require_admin() {
        Ok(_) => {
            storage::clear_bytes(FACE_DETECTION_FILE);
            CanisterResponse::Ok(())
        }
        Err(e) => CanisterResponse::Err(e),
    }
}

#[ic_cdk::update]
fn clear_face_recognition_model_bytes() -> CanisterResponse<()> {
    match require_admin() {
        Ok(_) => {
            storage::clear_bytes(FACE_RECOGNITION_FILE);
            CanisterResponse::Ok(())
        }
        Err(e) => CanisterResponse::Err(e),
    }
}

#[ic_cdk::update]
fn append_face_detection_model_bytes(bytes: Vec<u8>) -> CanisterResponse<()> {
    match require_admin() {
        Ok(_) => {
            storage::append_bytes(FACE_DETECTION_FILE, bytes);
            CanisterResponse::Ok(())
        }
        Err(e) => CanisterResponse::Err(e),
    }
}

#[ic_cdk::update]
fn append_face_recognition_model_bytes(bytes: Vec<u8>) -> CanisterResponse<()> {
    match require_admin() {
        Ok(_) => {
            storage::append_bytes(FACE_RECOGNITION_FILE, bytes);
            CanisterResponse::Ok(())
        }
        Err(e) => CanisterResponse::Err(e),
    }
}

#[ic_cdk::update]
fn setup_models() -> CanisterResponse<()> {
    match require_admin() {
        Ok(_) => {
            match setup(
                storage::bytes(FACE_DETECTION_FILE),
                storage::bytes(FACE_RECOGNITION_FILE),
            ) {
                Ok(_) => CanisterResponse::Ok(()),
                Err(err) => CanisterResponse::Err(format!("Failed to setup model: {}", err)),
            }
        }
        Err(e) => CanisterResponse::Err(e),
    }
}

#[ic_cdk::init]
fn init() {
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
}

#[ic_cdk::post_upgrade]
fn post_upgrade() {
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
}

#[derive(CandidType, Deserialize)]
pub enum CanisterResponse<T> {
    Ok(T),
    Err(String),
}

#[ic_cdk::query]
fn is_authorized() -> Result<(), String> {
    let caller = ic_cdk::caller();
    let admin =
        Principal::from_text(ADMIN_PRINCIPAL).map_err(|_| "Invalid admin principal".to_string())?;

    if caller != admin {
        Err("Unauthorized: only the admin can perform this action".to_string())
    } else {
        Ok(())
    }
}

#[ic_cdk::query]
fn get_recognition_result(user: Principal) -> Option<RecognitionResult> {
    RECOGNITION_RESULTS.with(|results| {
        results
            .borrow()
            .get(&user)
            .map(|(label, score)| RecognitionResult {
                label: label.clone(),
                score: *score,
            })
    })
}

fn require_admin() -> Result<(), String> {
    let caller = ic_cdk::caller();
    let admin =
        Principal::from_text(ADMIN_PRINCIPAL).map_err(|_| "Invalid admin principal".to_string())?;

    if caller != admin {
        Err("Unauthorized: only the admin can perform this action".to_string())
    } else {
        Ok(())
    }
}

#[ic_cdk::query]
fn get_add_callers() -> (u64, Vec<Principal>) {
    let callers =
        ADD_CALLERS.with(|callers| callers.borrow().iter().cloned().collect::<Vec<Principal>>());
    let count = callers.len() as u64;
    (count, callers)
}

#[ic_cdk::query]
fn get_all_recognition_results() -> Vec<String> {
    RECOGNITION_RESULTS.with(|results| {
        results
            .borrow()
            .iter()
            .map(|(principal, (label, score))| {
                format!(
                    "principal: {}, label: {}, score: {}",
                    principal, label, score
                )
            })
            .collect()
    })
}

#[ic_cdk::query]
fn get_cycles() -> u64 {
    api::canister_balance()
}

#[ic_cdk::update]
fn add_cycles() {
    let available_cycles = api::call::msg_cycles_available();
    let accepted_cycles = api::call::msg_cycles_accept(available_cycles);
    ic_cdk::println!("Accepted {} cycles", accepted_cycles);
}
