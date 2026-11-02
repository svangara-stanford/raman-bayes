# Phase 1 test list

Status: approved 2026-10-29. This is the list of tests Phase 1 commits to, mapped to the DESIGN.md §11 bullet each one satisfies. A test on this list may not be removed, skipped, xfailed or weakened without a hard stop (CLAUDE.md, Batch mode). Tests may be added; an addition is recorded in the sub-step's DEVLOG entry. Tests added after approval: two with patch 05, one with the A3 decision and one with the A9/A10 decision (marked below); `test_readout_mode_per_file` was dropped with A3 before any code existed.

Files are under `tests/unit/` unless they say `tests/integration/`. A parametrized test counts as one name.

Property tests that touch the filesystem or spawn processes set `@settings(deadline=None)` on that test only, with a comment saying why. The Hypothesis profile is not changed globally (decided 2026-10-29). This changes timing, not what a test asserts.

## Key

Unit: **U1** Info.txt parser · **U2** spectrum reader · **U3** filename regex · **U4** grid completeness · **U5** QC · **U6** PriorsSpec rules 1–8, `auto`, `derived:` · **U7** legacy CSV round-trip · **U8** Zarr round-trip, `dataset_id`, idempotent ingest · **U9** chunking · **U10** seed derivation · **U11** preprocessing · **U12** shard writer · **U13** lock · **U14** config · **U15** catalog state machine.

Integration: **I1** fixture end to end · **I2** seed invariance · **I3** kill and resume · **I4** concurrent duplicate · **I5** all-null merge · **I6** partial collect · **I7** CLI `--json` schemas and exit codes.

Other: **Fx** the §11 Fixtures paragraph · **LH** "legacy-csv header matches" (§13 done-when) · **SM** full-data smoke (§11, §13) · `§n` = no §11 bullet; the test enforces DESIGN §n or text added by one of the Phase 1 DESIGN patches: P1 ScanInfo (§3, §4), P2 SmoothingSpec (§5, §7.2), P3 RunConfig/RunPlan (§7.1, §9), P4 TaskState/SubmissionRecord and the transition table (§7.4, §8.5), P5 two-file shards (§7.2, §8.1).

## Batch A: `phase-1-01-foundation`

**01 Fixtures**: `test_make_fixture.py`, `test_fixtures.py`

| Test | Maps to |
|---|---|
| `test_selects_exactly_the_corner_files` | Fx |
| `test_copies_are_byte_identical` | Fx |
| `test_collects_all_five_info_files` | Fx |
| `test_copies_two_nmc111_spectra_of_different_sizes` | Fx, U2 |
| `test_refuses_to_overwrite_differing_fixture_without_force` | Fx |
| `test_force_overwrites_differing_fixture` | Fx |
| `test_rerun_on_identical_fixture_is_a_noop` | Fx |
| `test_missing_source_file_is_a_clear_error` | Fx |
| `test_mini_has_info_plus_25_spectra` | Fx |
| `test_mini_filenames_match_filename_re_within_corner` | Fx, U3 |
| `test_mini_spectra_are_1024_crlf_two_column_lines` | Fx, U2 |
| `test_mini_info_equals_info_txt_copy` | Fx |
| `test_five_info_files_present` | Fx, U1 |
| `test_info_files_keep_crlf_and_single_space_blank_lines` | Fx |
| `test_info_line_counts_are_102_for_nmc111_and_104_for_nmc811` | Fx, P1 |
| `test_nmc111_spectra_differ_in_byte_size` | Fx, U2 |
| `test_gitattributes_marks_fixtures_and_legacy_csvs_minus_text` | Fx |
| `test_precommit_excludes_fixtures_from_whitespace_fixers` | Fx |

**02 Nanofinder I/O and ScanInfo**: `test_filename.py`, `test_nanofinder_info.py`, `test_nanofinder_spectrum.py`, plus additions to `test_dataset.py`

| Test | Maps to |
|---|---|
| `test_parse_valid_names` (param.) | U3 |
| `test_parse_rejects_malformed_names` (param.) | U3 |
| `test_info_file_is_not_a_spectrum_name` | U3 |
| `test_custom_pattern_for_another_instrument` | U3 |
| `test_custom_pattern_missing_a_named_group_is_rejected` | U3 |
| `test_leading_zero_index_parses_to_same_integer` | U3, U4 |
| `test_grid_dimensions_per_file` (5 files: 50×25×1, 50×50×1) | U1 |
| `test_frame_dimentions_wins_over_frame_stage_for_z` | U1 |
| `test_n_accum_per_file` (1, 5, 5, 5, 10) | U1 |
| `test_n_series_is_1_in_all_files` | U1 |
| `test_n_series_2_copy_is_refused_as_time_series` | U1 |
| `test_exposure_read_from_line_shared_with_cycle_time` | U1 |
| `test_n_accum_read_from_line_shared_with_acquisition_mode` | U1 |
| `test_center_row_and_track_height_not_required` *(added with A3)* | U1 [A3] |
| `test_header_names` (system, detector, positioner) | U1 |
| `test_acquired_at_and_duration` | U1 |
| `test_twelve_hour_clock_across_noon` (NMC811_v1: 10:30:50 AM → 12:38:28 PM) | U1, P1 |
| `test_negative_duration_is_an_error` | U1, P1 |
| `test_channels_preamble_is_not_a_subsection` | U1 |
| `test_raw_stage_extent_and_lf_endings` | U1, P1 |
| `test_lf_and_crlf_info_parse_identically` | U1 |
| `test_unknown_keys_are_ignored` | U1, P1 |
| `test_missing_required_key_names_section_and_key` | U1, P1 |
| `test_scan_name_from_info_filename` | U1, P1 |
| `test_real_info_scan_info_json_round_trip` | U1 |
| `test_reads_nmc811_spectrum` (1024 rows, first and last axis values) | U2 |
| `test_reads_nmc111_spectra_of_differing_byte_sizes` | U2 |
| `test_lf_and_crlf_spectra_give_identical_arrays` | U2 |
| `test_fixture_spectra_have_no_bom` | U2 |
| `test_utf8_bom_is_tolerated` | U2 |
| `test_wrong_row_count_names_the_file` | U2, P1 |
| `test_non_numeric_line_names_file_and_line` | U2 |
| `test_axis_within_tolerance_matches` (0.049 cm⁻¹) | U2 |
| `test_axis_beyond_tolerance_is_a_mismatch` (0.051 cm⁻¹) | U2 |
| `test_scan_info_is_not_provisional` *(replaces a Phase 0 test, §2.4)* | P1 |
| `test_grid_counts_must_be_positive` | P1 |
| `test_single_point_axis_needs_zero_step_and_equal_ends` | P1 |
| `test_multi_point_axis_stop_within_half_step` | P1 |
| `test_acquisition_constraints` (exposure > 0, n_accum ≥ 1, central_pixel ≥ 0) | P1 |
| `test_n_series_other_than_1_rejected_by_model` | U1, P1 |
| `test_duration_must_be_non_negative` | P1 |
| `test_n_spectra_must_equal_grid_unless_partial` | P1 |

**03 Priors rules and `validate-priors`**: `test_priors_rules.py`, `test_validate.py`, `test_cli_validate_priors.py`

| Test | Maps to |
|---|---|
| `test_design_example_is_valid_without_warnings` | U6 |
| `test_rule1_unknown_schema_version_rejected` | U6 |
| `test_rule1_id_pattern` (param.: windows, peaks, end members) | U6 |
| `test_rule1_ids_unique_across_file` | U6 |
| `test_rule2_center_bounds_outside_window_rejected` | U6 |
| `test_rule2_center_bound_near_window_edge_warns` | U6 |
| `test_rule3_lower_not_below_upper_rejected` (Hypothesis) | U6 |
| `test_rule3_center_mean_outside_bounds_rejected` (Hypothesis) | U6 |
| `test_rule3_non_positive_fwhm_bounds_rejected` (Hypothesis) | U6 |
| `test_rule3_fwhm_upper_above_window_width_rejected` | U6 |
| `test_rule3_valid_bounds_accepted` (Hypothesis) | U6 |
| `test_rule4_overlapping_windows` (param.: neither, one, both `allow_overlap`) | U6 |
| `test_rule5_overlapping_centers_warn_unless_ordered` | U6 |
| `test_rule6_unknown_end_member_peak_rejected` | U6 |
| `test_rule6_peak_in_two_end_members_rejected` | U6 |
| `test_rule7_negative_margin_rejected` | U6 |
| `test_rule7_even_despike_kernel_rejected` | U6 |
| `test_rule7_smoothing_window_and_polyorder` (param.: even, < 3, polyorder ≥ window, negative polyorder) | U6, P2 |
| `test_all_errors_reported_not_only_the_first` | U6 |
| `test_auto_rejected_through_validate_file` | U6 |
| `test_derived_rejected_through_validate_file` | U6 |
| `test_quarter_range_sd_and_explicit_sd_wins` | §5 |
| `test_priors_hash_is_sha256_of_canonical_json` | §5 r9 |
| `test_priors_hash_ignores_yaml_formatting_and_key_order` | §5 r9 |
| `test_priors_hash_changes_with_any_value` | §5 r9 |
| `test_unsafe_yaml_tag_rejected` | §5 |
| `test_cli_valid_file_exits_0_with_hash` | I7 |
| `test_cli_invalid_file_exits_2_listing_every_error` | I7 |
| `test_cli_warnings_to_stderr_and_in_envelope` | I7, §5 |
| `test_cli_missing_file_exits_2` | I7 |
| `test_cli_yaml_syntax_error_exits_2_with_line` | I7 |
| `test_cli_messages_cite_no_repo_paths` | CLAUDE.md |

**04 Legacy CSV and `nmc_commercial.yaml`**: `test_legacy_csv.py`, `test_nmc_commercial.py`, `test_cli_priors_csv.py`

| Test | Maps to |
|---|---|
| `test_import_lithiation_csv` (BOM, CRLF, ints) | U7 |
| `test_import_carbon_csv` (ASCII, floats, `D`/`G`) | U7 |
| `test_import_column_mapping` | U7 |
| `test_import_lf_and_crlf_identical` | U7 |
| `test_import_with_and_without_bom_identical` | U7 |
| `test_import_lowercases_ids_and_keeps_label` | U7 [A9] |
| `test_import_rejects_malformed_row_naming_it` | U7 |
| `test_round_trip_import_export_import` (both files) | U7 |
| `test_export_import_export_is_byte_identical` (both files) | U7 |
| `test_export_header_exact` | U7 |
| `test_export_matches_original_rows_modulo_case` *(added with A9/A10)* | U7 [A9] |
| `test_export_uses_lf_and_no_bom` | U7 |
| `test_imported_fragment_passes_validation` | U6, U7 |
| `test_nmc_commercial_is_valid_and_approved` | §5, §14 |
| `test_nmc_commercial_equals_section5_example` | §5 |
| `test_nmc_commercial_peaks_match_legacy_csvs` | §14 |
| `test_cli_import_csv_prints_fragment` / `--json` | I7 |
| `test_cli_import_csv_bad_range_exits_2` | I7 |
| `test_cli_export_csv_prints_csv` / `--json` | I7 |
| `test_cli_export_csv_unknown_window_exits_2` | I7 |

**05 Config**: `test_config.py`, `test_site_profiles.py`

| Test | Maps to |
|---|---|
| `test_expands_braced_var` | U14 |
| `test_unset_var_error_names_variable_and_key` | U14 |
| `test_only_braced_syntax_is_expanded` (`$VAR` stays literal; unclosed `${` is an error) | U14 |
| `test_several_vars_in_one_value` | U14 |
| `test_expansion_in_nested_values_and_lists` | U14 |
| `test_env_override_values_are_not_expanded` | U14, P3 |
| `test_precedence_defaults_profile_env_cli` (param. per layer) | U14 |
| `test_nested_env_override_double_underscore` | U14, P3 |
| `test_profile_name_resolves_packaged_first` | §9 |
| `test_profile_name_falls_back_to_cwd_configs_profiles` | §9 |
| `test_profile_path_used_as_given` | §9 |
| `test_raman_profile_env_selects_profile` | §9 |
| `test_unknown_profile_error_names_searched_locations` | §9 [A15] |
| `test_unknown_profile_key_rejected_naming_it` | P3 |
| `test_relative_paths_resolve_against_cwd` | P3 |
| `test_resolved_config_has_absolute_paths_and_profile_source` | P3 |
| `test_packaged_local_profile_validates` | §9 |
| `test_site_profiles_validate_with_env_set` (labpc, sherlock) | §9 |
| `test_sherlock_profile_without_scratch_names_scratch` | U14 |
| `test_site_profiles_match_design_md` | §9 |

**06 Catalog**: `test_catalog.py`, `test_cli_catalog_lists.py`

| Test | Maps to |
|---|---|
| `test_tables_and_columns_match_section_8_5` | §8.5 |
| `test_rollback_journal_not_wal` | §8.5 |
| `test_user_version_is_1` | §8.5 [A16] |
| `test_every_legal_transition_succeeds` (param. over the P4 table) | U15 |
| `test_every_illegal_transition_raises` (param. over the other 38 of the 49 ordered pairs, self-transitions included) | U15 |
| `test_transition_from_stale_expected_status_raises` | U15 |
| `test_refresh_applies_transitions_in_order` | U15, P4 |
| `test_status_columns_hold_enum_values` | P4 |
| `test_dataset_insert_and_get_round_trip_manifest` | §8.5 |
| `test_duplicate_dataset_insert_reports_exists` | U8 |
| `test_priors_spec_insert_idempotent_by_hash` | §8.5 |
| `test_run_and_task_rows_created_atomically` | P3 |
| `test_task_row_updated_from_task_state` | P4 |
| `test_catalog_created_with_parent_dirs_on_first_write` | §8.5 |
| `test_cli_datasets_list_empty_exits_0` / `--json` | I7 |
| `test_cli_datasets_show_unknown_exits_2` | I7 |
| `test_cli_runs_list_and_show_json` | I7 |
| `test_cli_runs_show_unknown_exits_2` | I7 |

## Batch B: `phase-1-07-pipeline`

**07 Zarr and ingest**: `test_zarr_dataset.py`, `test_grid.py`, `test_qc.py`, `test_ingest.py`, `test_cli_ingest.py`

| Test | Maps to |
|---|---|
| `test_round_trip_bit_identical` | U8 |
| `test_arrays_dtypes_shapes_chunks_shards_codecs` | §4 |
| `test_spectrum_id_is_raster_order` | §4 |
| `test_coordinates_from_frame_dimentions` | §3 |
| `test_attrs_repeat_manifest_values` | P1 |
| `test_reader_rejects_attrs_disagreeing_with_manifest` | P1 |
| `test_count_at_or_above_2_pow_24_refused` | §4 |
| `test_content_hash_matches_definition` | U8, P1 |
| `test_dataset_id_deterministic_and_location_independent` | U8 |
| `test_dataset_id_changes_with_one_count` | U8 |
| `test_complete_grid_ok` | U4 |
| `test_missing_file_error` | U4 |
| `test_duplicate_index_error` | U4 |
| `test_out_of_grid_file_error` | U4 |
| `test_three_grid_errors_are_distinct` | U4 |
| `test_allow_partial_records_missing_in_mask` | U4 |
| `test_allow_partial_still_rejects_duplicate_and_out_of_grid` | U4 |
| `test_scan_name_mismatch_names_first_file` | P1 |
| `test_saturated_at_n_accum_times_65535` (param. n_accum 1, 5, 10) | U5 |
| `test_just_below_threshold_not_saturated` | U5 |
| `test_flat_top_of_three_sets_saturated` | U5 |
| `test_two_equal_samples_at_max_not_flat_top` | U5 |
| `test_tall_legitimate_peak_not_flagged` | U5 |
| `test_nan_or_inf_flag` | U5 |
| `test_negative_flag` | U5, §3 |
| `test_axis_mismatch_flag` | U2, §3 |
| `test_flags_combine_in_uint16` | §3 |
| `test_ingest_fixture_allow_partial` (25 spectra, 5×5 mask) | U4, U8 |
| `test_ingest_fixture_without_allow_partial_fails_count_check` | §3 r3 |
| `test_second_ingest_is_noop_exit_3` | U8 |
| `test_ingest_registers_dataset_in_catalog` | §8.5 |
| `test_ingest_parent_folder_ingests_each_scan` | §10 |
| `test_dry_run_reports_and_writes_nothing` | §10 |
| `test_axis_mismatch_fails_unless_allowed` | U2, §3 |
| `test_failed_ingest_leaves_no_store` | §4 [B7] |
| `test_n_series_2_exits_2_with_time_series_message` | U1 |
| `test_cli_ingest_json_and_exit_codes` | I7 |

**08 Preprocessing**: `test_preprocessing.py`, plus a change to `test_priors.py`

| Test | Maps to |
|---|---|
| `test_clip_includes_margin` | U11 |
| `test_clip_with_zero_margin` | U11 |
| `test_single_channel_spike_removed` | U11 |
| `test_adjacent_real_peak_preserved` | U11 |
| `test_despike_none_is_identity` | §7.2 |
| `test_als_flattens_sloped_baseline_within_tolerance` | U11 |
| `test_als_keeps_peak_height_within_tolerance` | U11 |
| `test_savgol_uses_channel_window_and_interp_mode` | P2 |
| `test_smoothing_none_is_identity` | P2 |
| `test_clamp_nonnegative` / `test_clamp_false_keeps_negatives` | §7.2 |
| `test_trim_returns_exactly_window_channels` | U11 |
| `test_trim_on_non_uniform_axis` | U11 |
| `test_noise_sd_is_scaled_mad_of_median9_residual` | §7.2 [B3] |
| `test_noise_sd_of_white_noise_near_sigma` (seeded) | §7.2 |
| `test_steps_run_in_section_7_2_order` | §7.2 |
| `test_prepare_returns_float64_prepared_spectrum` | §6 |
| `test_preprocessing_is_deterministic` | U10 |
| `test_smoothing_spec_is_not_provisional` *(replaces a Phase 0 test, §2.4)* | P2 |

**09 Mock engine, registry, seeds**: `test_engine_registry.py`, `test_mock_engine.py`, `test_seeding.py`

| Test | Maps to |
|---|---|
| `test_mock_registered_in_entry_point_group` | §6 |
| `test_unknown_engine_error_lists_available` | §6 |
| `test_loaded_engine_satisfies_peak_engine` | §6 |
| `test_wheel_declares_engine_entry_point` | §6, §2 |
| `test_name_and_version` (mock, 0.1.0) | §6 |
| `test_center_is_argmax_within_bounds` | §6 |
| `test_fwhm_is_bounds_midpoint` | §6 |
| `test_area_is_sum_times_channel_width` | §6 [B13] |
| `test_height_is_max_within_bounds` | §6 |
| `test_eta_is_one_half` | §6 |
| `test_presence_formula_with_seeded_jitter` | §6 |
| `test_presence_clipped_to_unit_interval` | §6 |
| `test_absent_peaks_have_none_summaries` | §6, I1 |
| `test_sd_five_percent_and_quantiles_1_96_sd` | §6 |
| `test_presence_is_scale_free` (×3 intensity and noise) | §6, SM |
| `test_zero_noise_sd_does_not_divide_by_zero` | §6 [B4] |
| `test_deterministic_for_a_seed` | §6 |
| `test_never_calls_np_random_seed` | §6 |
| `test_fit_many_equals_fit_loop` | §6 |
| `test_spawn_key_is_window_index_and_spectrum_id` | U10 |
| `test_seeds_independent_of_task_boundaries` (Hypothesis partitions) | U10 |
| `test_seeds_independent_of_sub_batching_and_retry` | U10 |
| `test_distinct_spectra_get_distinct_streams` | U10 |

**10 Plan**: `test_chunking.py`, `test_uuid7.py`, `test_plan.py`, `test_cli_plan.py`, plus additions to `test_run.py`

| Test | Maps to |
|---|---|
| `test_explicit_size_partitions_exactly` (Hypothesis n, s) | U9 |
| `test_derived_size_boundaries_multiples_of_1024` (Hypothesis n, workers) | U9 |
| `test_derived_size_raised_to_respect_cap` (Hypothesis) | U9 |
| `test_explicit_size_over_cap_refused` | U9 |
| `test_last_task_may_be_short` | U9 |
| `test_n_below_size_gives_one_task` | U9 |
| `test_uuid7_version_and_variant_bits` | §7.1 |
| `test_uuid7_timestamp_is_unix_ms` | §7.1 |
| `test_uuid7_strictly_increasing_over_10000_calls` | §7.1 |
| `test_uuid7_unique` | §7.1 |
| `test_uuid7_canonical_lowercase_string` | §7.1 |
| `test_run_plan_tasks_must_partition_n_spectra` | U9, P3 |
| `test_run_plan_priors_hash_must_match` | P3 |
| `test_run_plan_run_id_must_be_uuid7` | P3 |
| `test_seed_range` (RunConfig and RunPlan) | P3 |
| `test_run_config_spectra_per_task_at_least_1` | P3 |
| `test_run_config_and_run_plan_not_provisional` *(replaces part of a Phase 0 test, §2.4)* | P3 |
| `test_plan_fixture_size_16_gives_2_tasks` | I1 |
| `test_plan_writes_run_plan_and_catalog_rows` | P3, §8.5 |
| `test_plan_freezes_absolute_resolved_config` | P3 |
| `test_rule8_window_under_32_channels_exits_2` | U6 |
| `test_rule8_margin_outside_axis_exits_2` | U6 |
| `test_rule8_segment_shorter_than_smoothing_window_exits_2` | U6, P2 |
| `test_seed_drawn_and_recorded_when_absent` | P3 |
| `test_explicit_seed_kept` | P3 |
| `test_unknown_dataset_exits_2` | I7 |
| `test_engine_validate_messages_refuse_plan` | §6 |
| `test_plan_registers_priors_spec` | §8.5 |
| `test_draft_priors_plan_warns` | §5 [B17] |
| `test_cli_plan_json_lists_tasks` | I7 |

## Batch C: `phase-1-11-execution`

**11 Shards and run-task**: `test_shards.py`, `test_run_task.py`, `test_cli_run_task.py`

| Test | Maps to |
|---|---|
| `test_final_part_absent_until_replace` | U12 |
| `test_interrupted_write_leaves_no_final_file` | U12 |
| `test_leftover_tmp_ignored_and_replaced_on_resume` | U12 |
| `test_all_null_column_stays_float32` | U12 |
| `test_parts_written_with_explicit_schemas` | U12, §8.2 |
| `test_fsync_before_replace` | U12 |
| `test_valid_part_skipped_on_resume` | §7.2 |
| `test_invalid_part_rewritten_on_resume` (both files rewritten) | §7.2, P5 |
| `test_sub_batch_valid_only_with_both_files` *(added with patch 05)* | §7.2, P5 [C4] |
| `test_fits_written_before_peaks` *(added with patch 05)* | §7.2, P5 [C4] |
| `test_lock_is_o_excl_json_host_pid_started_at` | U13 |
| `test_fresh_lock_exits_3` | U13 |
| `test_stale_lock_taken_over` | U13 |
| `test_lock_stale_minutes_default_30_and_profile_override` | U13 |
| `test_done_marker_contents` | §7.2 |
| `test_done_marker_atomic_and_lock_removed` | §7.2 |
| `test_done_task_exits_3_untouched` | §7.2 [C3] |
| `test_sub_batches_of_512` | §7.2 |
| `test_nan_and_saturated_skipped_without_fit` | §7.2 |
| `test_negative_and_axis_mismatch_still_fitted` | §7.2 [B5] |
| `test_per_spectrum_exception_becomes_failed_row` | §7.2 |
| `test_missing_store_exits_4` | §7.2, I7 |
| `test_engine_version_mismatch_exits_4` | P3 |
| `test_never_touches_catalog` | §7.2, §15 #5 |
| `test_logs_to_task_log` | §7.2 |
| `test_absent_peaks_null_in_rows` | §15 #6 |
| `test_end_member_id_mapped_or_null` | §8.2 |
| `test_workers_gt_1_identical_parts` | U10 [C2] |
| `test_cli_run_task_exit_codes` (0, 3, 4) | I7 |

**12 LocalBackend, submit, status, cancel, run**: `test_local_backend.py`, `test_cli_submit_status.py`, plus additions to `test_run.py`

| Test | Maps to |
|---|---|
| `test_task_state_done_iff_marker` | P4 |
| `test_task_state_counts_only_when_done` | P4 |
| `test_task_state_attempts_non_negative` | P4 |
| `test_submission_task_indices_non_empty_increasing` | P4 |
| `test_submission_slurm_id_iff_slurm_and_not_dry_run` | P4 |
| `test_submission_attempt_at_least_1` | P4 |
| `test_task_state_and_submission_not_provisional` *(replaces part of a Phase 0 test, §2.4)* | P4 |
| `test_submit_runs_all_tasks_in_spawn_pool` | §7.3 |
| `test_submit_returns_record_attempt_1` | P4 |
| `test_submission_json_is_array_in_attempt_order` | P4 |
| `test_dry_run_writes_nothing_changes_nothing` | P4 |
| `test_poll_maps_markers_to_states` (pending, running, done, failed, stale lock) | P4 |
| `test_poll_done_marker_wins` | P4, §7.3 |
| `test_cancel_terminates_running_tasks` | §7.3 [C7] |
| `test_resubmit_only_tasks_without_done` | §7.3 |
| `test_resubmit_refuses_task_after_3_attempts` | P4 |
| `test_failure_with_nothing_in_flight_marks_run_failed` | P4 [A5] |
| `test_cli_submit_planned_run_completes` | I7 |
| `test_cli_submit_state_refusals_exit_codes` | I7, §7.4 |
| `test_cli_submit_dry_run_with_confirm_exits_2` | §10 |
| `test_cli_status_json_lists_task_states` | I7 |
| `test_cli_status_failures_lists_failed_tasks` | §10 |
| `test_cli_status_watch_returns_on_terminal_run` | §10 |
| `test_cli_cancel_requires_confirm` | §10 |
| `test_cli_cancel_planned_run` | U15 |
| `test_cli_run_plans_submits_and_waits` | §10 |
| `test_cli_run_refuses_non_local_backend` | §10 |

**13 Collect**: `test_merge.py`, `test_run_manifest.py`, `test_collect.py`

| Test | Maps to |
|---|---|
| `test_merge_rejects_schema_mismatch` | §8.2 |
| `test_merge_all_null_shard_keeps_float32` | I5, U12 |
| `test_peaks_sorted_with_64k_row_groups` | §8.2 |
| `test_fits_diag_columns_unioned_as_float32` | §8.2 |
| `test_kv_metadata_has_manifest_and_schema_version` | §8.2 |
| `test_dictionary_columns_preserved` | §8.2 |
| `test_merge_rejects_duplicate_keys` | I3 |
| `test_shard_hashes_checked_against_done_markers` | §8.3 |
| `test_manifest_has_all_section_8_3_keys` | I1, §8.3 |
| `test_manifest_counts_failed_and_skipped_from_fits` | §8.3, P4 |
| `test_manifest_provenance_from_done_markers` | §8.3 [C6] |
| `test_mixed_code_versions_refused` | §8.3 [C6] |
| `test_collect_refuses_incomplete_run` | I6 |
| `test_collect_partial_marks_manifest_keeps_status` | I6, P4 |
| `test_collect_complete_moves_to_collected` | U15 |
| `test_collect_twice_exits_3` | §10 |
| `test_collect_copies_to_archive_root_when_set` | §7.5 [C14b] |
| `test_cli_collect_as_job_on_local_exits_2` | §10 [B8] |

**14 Export, runs import, doctor** (ends in a planned stop for the ci.yml change, see the last row of the Phase 0 table): `test_export.py`, `test_runs_import.py`, `test_doctor.py`

| Test | Maps to |
|---|---|
| `test_legacy_csv_header_matches_committed_string` (per window) | LH, Fx [C11] [A9] |
| `test_legacy_csv_one_file_per_window_in_yaml_peak_order` | §8.4 |
| `test_legacy_csv_min_presence` | §8.4 |
| `test_legacy_csv_absent_fill_default_zero` | §8.4 |
| `test_legacy_csv_height_source` | §8.4 |
| `test_legacy_csv_coords_index_default_and_um` | §8.4, §15 #9 |
| `test_legacy_csv_window_option` | §8.4 |
| `test_csv_export_flattens_peaks_parquet` | §8.4 |
| `test_csv_export_nulls_are_empty_fields` | §15 #6 |
| `test_export_uncollected_run_exits_2` | I7 [C13] |
| `test_duckdb_query_over_results` | §8.4 |
| `test_runs_import_registers_run` | §8.5 |
| `test_runs_import_existing_exits_3` | I7 |
| `test_runs_import_invalid_dir_exits_2` | I7 |
| `test_runs_import_unknown_dataset_exits_2` | §8.5 [C14] |
| `test_doctor_local_ok_exits_0` | §10 |
| `test_doctor_reports_profile_paths_and_engines` | §10 |
| `test_doctor_unresolved_var_exits_2` | U14 |
| `test_doctor_skips_slurm_checks_on_local` | §10 |
| `test_doctor_warns_dataset_older_than_60_days` | §7.5 |
| `test_doctor_json_validates` | I7 |

**15 Integration and `--json` gaps**: `tests/integration/test_end_to_end.py`, `test_resilience.py`, `test_cli_json.py`; `tests/unit/test_readme.py`

| Test | Maps to |
|---|---|
| `test_e2e_fixture_pipeline` (ingest → validate → plan 2 tasks → local → collect → both exports; 100 peak rows; explicit schema; manifest hashes) | I1 |
| `test_e2e_absent_peaks_null_not_zero` (null ⇔ presence < 0.5) | I1 |
| `test_e2e_synthetic_flat_spectra_give_null_rows` | I1 |
| `test_seed_invariance_1024_vs_8` | I2 |
| `test_seed_invariance_workers_1_vs_4` | I2 |
| `test_kill_and_resume_identical_no_duplicates` | I3 |
| `test_concurrent_duplicate_one_exits_3_results_intact` | I4 |
| `test_merge_with_all_null_shard` | I5 |
| `test_collect_incomplete_refuses_partial_succeeds` | I6 |
| `test_every_command_json_validates_against_schema` (jsonschema) | I7 |
| `test_exit_codes_per_section_10` (param. scenarios for 0/2/3/4) | I7 |
| `test_usage_error_under_json_emits_envelope_exit_2` | I7 (plan §3.3) |
| `test_unhandled_exception_exits_4` | I7 (plan §3.3) |
| `test_no_ansi_when_not_a_tty` | §10 |
| `test_readme_documents_full_data_smoke` | SM |

The smoke run itself (both archives, 5 datasets, 11,250 spectra, < 3 min, NMC111 vs NMC811 presence distributions) is manual. It is run at the end of 15, with the numbers recorded in the DEVLOG and the PR body.

## Phase 0 tests that change (approved 2026-10-29)

Approved as a block on two conditions: each change is recorded in the DEVLOG entry of the sub-step that makes it, and no test outside this table is changed. Any other deletion, skip, xfail or weakening is a hard stop (CLAUDE.md).

| Test (file) | Sub-step | Change | Why |
|---|---|---|---|
| `test_scan_info_is_marked_provisional` (test_dataset) | 02 | replaced by `test_scan_info_is_not_provisional` | finalized |
| `axes`, `acqs`, `manifests` strategies (test_dataset) | 02 | generate only values the new validators accept (consistent axes, exposure > 0, n_spectra ≤ grid, duration ≥ 0) | P1 validators |
| `test_manifest_rejects_unknown_nested_key` (test_dataset) | 02 | fixture grid `ny, nz` 1 → 2; assertion unchanged | P1 axis rule rejects the old fixture (1-point axes with step 1); approved 2026-11-02 |
| `test_smoothing_spec_is_marked_provisional` (test_priors) | 08 | replaced by `test_smoothing_spec_is_not_provisional` | finalized |
| `MODEL_KEYS` RunPlan and SubmissionRecord entries (test_run) | 10, 12 | new key sets per P3 and P4 | P3, P4 |
| `PROVISIONAL` and `test_provisional_marker` (test_run) | 10, 12 | entries removed as each model is finalized; replaced by the `*_not_provisional` tests above. An empty parametrize would skip, so the test is deleted at 12 when its list is empty. | finalized |
| `test_optional_defaults` (test_run) | 10 | RunConfig part unchanged; TaskState part unchanged | — |
| `plans`, `submissions`, `task_states` strategies (test_run) | 10, 12 | seed < 2⁶³; valid task partitions; priors included; submission and task-state invariants respected | P3, P4 |
| `test_run_config_json_round_trip` (test_run) | 10 | unchanged unless the seed or path changes | — |
| Dummy backend and plan constructors (test_ports, typing_checks_ports) | 10, 12 | pass the new required fields | P3, P4 |
| `test_stub_exits_nothing_to_do`, `test_stub_json_envelope` (test_cli_commands) | 03–14 | parametrized over the commands that are still stubs; each sub-step removes the commands it implements; both tests are deleted at 14 when none remain | real commands |
| `test_profile_is_accepted` (test_cli_commands) | 05 (applied at 03) | asserts `--profile` parses (no Click usage error) rather than exit 3, because loading `sherlock` without `SCRATCH` is now a real exit-2 error | config is real; `validate-priors` exits 2 from 03 |
| `test_emit_human_success_goes_to_stdout` (test_cli_output) | 05 | warnings on stderr, message alone on stdout | warnings to stderr for every command; approved 2026-11-02 |
| `test_run_task_workers` ("4" → 3), `test_export_accepts_all_section_8_4_options`, `test_ingest_accepts_all_flags` (test_cli_commands) | 07, 11, 14 | assert "not a usage error" instead of exit 3; each command's own tests cover real behaviour | real commands |
| CI `wheel-install` step `test "$code" -eq 3` for `doctor --json` (ci.yml, not a test file) | 14 | [C15] becomes `-eq 0`. Sub-step 14 is a planned stop: commit doctor, write the ci.yml diff to the scratchpad, print it, stop; Sreya applies it. A red wheel-install job at that point is expected and is not the "CI red twice" hard stop. | real doctor |

## Decision tags

A tag in square brackets marks a test whose expected behaviour depends on a Phase 1 plan-review decision. Approved decisions are recorded in the DEVLOG; while a tag is open, the test follows the recommendation below. If a DEVLOG entry records a different answer, the test follows the DEVLOG. Changing the test's assertion to match is then a planned change, not a weakening, and that DEVLOG entry records it.

| Tag | Decision | Status |
|---|---|---|
| A3 | `Readout Mode`, `Center Row` and `Track Height` are not stored; the parser must not require the last two. `test_readout_mode_per_file` and the `MANIFEST_KEYS` change were dropped and one test added | decided 2026-10-29 |
| A5 | `failed` = nothing pending or running and a task has no `_DONE.json`; resubmission capped at 3 attempts per task | decided 2026-10-29 |
| A9 | §5 ids are lowercase-only (`^[a-z][a-z0-9_]{0,31}$`), so legacy CSV import lowercases the first column into the id and keeps the original as `label` when it differs; export writes the id and float repr. The priors-CSV header has no peak names and stays an exact match. Peak names (`D`/`G` against `d`/`g`) compare modulo case: in priors-CSV rows (with numbers compared as values and BOM/CRLF normalised), and against an old counts CSV's header if one is found | decided 2026-10-29 |
| A15 | the unknown-profile error may name the absolute paths searched, including `configs/profiles` | decided 2026-10-29 |
| A16 | catalog `PRAGMA user_version = 1`, rollback journal, 30 s busy timeout, `BEGIN IMMEDIATE` writes | decided 2026-10-29 |
| B3 | `noise_sd` = 1.4826·MAD of the 9-point median-filter residual (`scipy.ndimage.median_filter`, `mode="nearest"`) | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| B4 | mock with `noise_sd == 0`: presence = 1 if height > 0, else 0, before jitter | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| B5 | only `nan_or_inf` and `saturated` skip fitting; `negative` = any raw count < 0 | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| B7 | ingest hashes in memory, exits 3 if the id exists, otherwise writes a temp store and renames it | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| B8 | `--as-job` and `--archive` exit 2 on local in Phase 1 | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| B13 | mock area = Σ yᵢ·\|Δνᵢ\| with `np.gradient`; one jitter draw per peak in YAML order | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| B17 | `plan` accepts `draft` priors with a warning in Phase 1 | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| C2 | `run-task --workers N` parallelises sub-batches within a task; output identical | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| C3 | run-task on a task with `_DONE.json` exits 3 and touches nothing | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| C4 | two files per sub-batch, `fits-<k>.parquet` then `peaks-<k>.parquet` (patch 05) | approved 2026-10-29 |
| C6 | provenance comes from `_DONE.json`; collect exits 4 if tasks ran different code | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| C7 | local cancel sends SIGTERM to fresh `_LOCK` pids; Ctrl-C keeps parts; resume via `submit --resubmit-failed` | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| C11 | legacy-csv headers use peak ids (`u1`, `u2`, `d`, `g`); if an old counts CSV turns up, its header is compared modulo case (A9) | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| C13 | exporting an uncollected run exits 2 | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| C14 | `runs import` needs the dataset in the catalog and registers the folder in place | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| C14b | `collect` copies results to `archive_root` when the profile sets it | approved 2026-10-29 (recommended answer; recorded in the sub-step's DEVLOG entry) |
| C15 | sub-step 14 is a planned stop for the ci.yml doctor change | approved 2026-10-29 |
