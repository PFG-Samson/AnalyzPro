"""Automated insights generation from analysis results."""

import numpy as np
from typing import Dict, List
from pathlib import Path
from ..utils import get_logger

logger = get_logger(__name__)


class InsightsGenerator:
    """Generates automated insights from analysis results."""
    
    @staticmethod
    def generate_ndvi_insights(ndvi_data: np.ndarray, stats: Dict) -> Dict:
        """
        Generate insights from NDVI analysis with vegetation stress detection.
        
        Args:
            ndvi_data: NDVI array
            stats: NDVI statistics
            
        Returns:
            Insights dictionary
        """
        insights = {
            'analysis_type': 'NDVI - Vegetation Health & Stress Analysis',
            'summary': '',
            'key_findings': [],
            'recommendations': [],
            'vegetation_categories': {}
        }
        
        mean_ndvi = stats.get('mean', 0)
        veg_cover = stats.get('vegetation_cover_percent', 0)
        healthy_veg = stats.get('healthy_vegetation_percent', 0)
        stress_pct = stats.get('potential_stress_percent', 0)
        stress_area = stats.get('stressed_area_km2', 0)
        vigor = stats.get('vegetation_vigor_mean', 0)
        
        # Summary
        if mean_ndvi > 0.5:
            health = "excellent"
        elif mean_ndvi > 0.3:
            health = "good"
        elif mean_ndvi > 0.1:
            health = "moderate"
        else:
            health = "poor"
        
        insights['summary'] = (
            f"The study area shows {health} overall vegetation health with a mean "
            f"NDVI of {mean_ndvi:.3f}. Approximately {veg_cover:.1f}% of the area "
            f"contains vegetation, with {healthy_veg:.1f}% classified as healthy. "
            f"Vegetation vigor index: {vigor:.3f}."
        )
        
        # Vegetation categories breakdown
        insights['vegetation_categories'] = {
            'No Vegetation': f"{stats.get('no_vegetation_percent', 0):.1f}%",
            'Sparse (Stressed)': f"{stats.get('sparse_vegetation_percent', 0):.1f}%",
            'Moderate': f"{stats.get('moderate_vegetation_percent', 0):.1f}%",
            'Healthy': f"{stats.get('healthy_vegetation_percent', 0):.1f}%",
            'Very Healthy': f"{stats.get('very_healthy_vegetation_percent', 0):.1f}%"
        }
        
        # Key findings
        if veg_cover > 70:
            insights['key_findings'].append(
                f"High vegetation coverage ({veg_cover:.1f}%) indicates well-vegetated area"
            )
        elif veg_cover < 30:
            insights['key_findings'].append(
                f"Low vegetation coverage ({veg_cover:.1f}%) suggests sparse vegetation or urban/bare land"
            )
        
        if healthy_veg > 50:
            insights['key_findings'].append(
                "More than half of vegetation shows healthy growth patterns"
            )
        elif healthy_veg < 20:
            insights['key_findings'].append(
                "Limited healthy vegetation may indicate stress factors (drought, disease, etc.)"
            )
        
        # Stress detection findings
        if stress_pct > 15:
            insights['key_findings'].append(
                f"⚠️ Significant vegetation stress detected: {stress_pct:.1f}% of area "
                f"({stress_area:.2f} km²) shows sparse/stressed vegetation patterns"
            )
            insights['recommendations'].append(
                "Priority investigation needed for stressed vegetation areas"
            )
            insights['recommendations'].append(
                "Check for: water stress, nutrient deficiency, pest/disease, or seasonal dormancy"
            )
        elif stress_pct > 5:
            insights['key_findings'].append(
                f"Moderate vegetation stress: {stress_pct:.1f}% of area may require monitoring"
            )
        
        if vigor < 0.4 and veg_cover > 20:
            insights['key_findings'].append(
                f"⚠️ Low vegetation vigor ({vigor:.3f}) across vegetated areas - investigate health issues"
            )
        
        water_pixels = np.sum(ndvi_data < 0)
        if water_pixels > ndvi_data.size * 0.1:
            water_pct = water_pixels / ndvi_data.size * 100
            insights['key_findings'].append(
                f"Significant water bodies detected ({water_pct:.1f}% of area)"
            )
        
        # Recommendations
        if mean_ndvi < 0.3:
            insights['recommendations'].append(
                "Consider vegetation enhancement or irrigation for improved green cover"
            )
        
        if healthy_veg < 30 and veg_cover > 30:
            insights['recommendations'].append(
                "Implement vegetation monitoring program to track health changes over time"
            )
        
        if vigor > 0.6:
            insights['recommendations'].append(
                "✅ Vegetation shows good vigor - maintain current land management practices"
            )
        
        logger.info("Generated enhanced NDVI insights with stress detection")
        return insights
    
    @staticmethod
    def generate_ndwi_insights(ndwi_data: np.ndarray, stats: Dict) -> Dict:
        """Generate insights from NDWI analysis."""
        insights = {
            'analysis_type': 'NDWI',
            'summary': '',
            'key_findings': [],
            'recommendations': []
        }
        
        water_cover = stats.get('water_cover_percent', 0)
        mean_ndwi = stats.get('mean', 0)
        
        insights['summary'] = (
            f"Water body analysis reveals {water_cover:.1f}% water coverage "
            f"with mean NDWI of {mean_ndwi:.3f}."
        )
        
        if water_cover > 50:
            insights['key_findings'].append(
                f"Study area is dominated by water bodies ({water_cover:.1f}%)"
            )
        elif water_cover > 20:
            insights['key_findings'].append(
                f"Significant water presence detected ({water_cover:.1f}%)"
            )
        elif water_cover > 5:
            insights['key_findings'].append(
                f"Moderate water features present ({water_cover:.1f}%)"
            )
        else:
            insights['key_findings'].append(
                f"Limited water features in study area ({water_cover:.1f}%)"
            )
        
        if water_cover > 30:
            insights['recommendations'].append(
                "Consider detailed water quality and depth analysis"
            )
        
        logger.info("Generated NDWI insights")
        return insights
    
    @staticmethod
    def generate_optical_insights(analysis_type: str, data: np.ndarray, 
                                 stats: Dict) -> Dict:
        """Generate insights from optical analysis (NDBI, EVI, SAVI, Classification)."""
        insights = {
            'analysis_type': analysis_type,
            'summary': '',
            'key_findings': [],
            'recommendations': []
        }
        
        mean_val = stats.get('mean', 0)
        
        if 'NDBI' in analysis_type:
            # NDBI - Built-up area analysis
            urban_pct = stats.get('built_up_percent', 0)
            insights['summary'] = (
                f"Built-up area analysis shows mean NDBI of {mean_val:.3f}. "
                f"Approximately {urban_pct:.1f}% of area classified as built-up."
            )
            
            if urban_pct > 50:
                insights['key_findings'].append(
                    f"Highly urbanized area ({urban_pct:.1f}% built-up)"
                )
            elif urban_pct > 25:
                insights['key_findings'].append(
                    f"Significant urban development present ({urban_pct:.1f}%)"
                )
            else:
                insights['key_findings'].append(
                    f"Limited urban areas ({urban_pct:.1f}% built-up)"
                )
        
        elif 'EVI' in analysis_type:
            # EVI - Enhanced Vegetation Index
            veg_cover = stats.get('vegetation_cover_percent', 0)
            insights['summary'] = (
                f"Enhanced Vegetation Index analysis shows mean EVI of {mean_val:.3f}. "
                f"Vegetation coverage: {veg_cover:.1f}%."
            )
            
            if mean_val > 0.4:
                insights['key_findings'].append(
                    "High vegetation density with good atmospheric correction"
                )
            elif mean_val > 0.2:
                insights['key_findings'].append(
                    "Moderate vegetation presence detected"
                )
        
        elif 'SAVI' in analysis_type:
            # SAVI - Soil Adjusted Vegetation Index
            veg_cover = stats.get('vegetation_cover_percent', 0)
            insights['summary'] = (
                f"Soil-Adjusted Vegetation Index analysis (mean: {mean_val:.3f}) "
                f"accounting for soil brightness effects. Vegetation coverage: {veg_cover:.1f}%."
            )
            
            if mean_val > 0.3:
                insights['key_findings'].append(
                    "Good vegetation cover with minimal soil brightness interference"
                )
            elif mean_val < 0.1:
                insights['key_findings'].append(
                    "Sparse vegetation or high soil exposure"
                )
        
        elif 'Classification' in analysis_type:
            # Land cover classification
            n_classes = stats.get('n_clusters', 0)
            class_dist = stats.get('class_distribution', {})
            
            insights['summary'] = (
                f"Land cover classification identified {n_classes} distinct classes."
            )
            
            if class_dist:
                # Handle both numeric values and nested dicts
                dist_items = []
                for k, v in class_dist.items():
                    if isinstance(v, dict):
                        # If v is a dict with 'percentage', extract it
                        pct = v.get('percentage', v.get('percent', 0))
                        dist_items.append(f"{k}: {pct:.1f}%")
                    elif isinstance(v, (int, float)):
                        dist_items.append(f"{k}: {v:.1f}%")
                    else:
                        dist_items.append(f"{k}: {v}")
                
                if dist_items:
                    insights['key_findings'].append(
                        f"Class distribution: {', '.join(dist_items)}"
                    )
            
            insights['recommendations'].append(
                "Review classification results for accuracy and refinement"
            )
        
        logger.info(f"Generated {analysis_type} insights")
        return insights
    
    @staticmethod
    def generate_sar_insights(analysis_type: str, data: np.ndarray, 
                             stats: Dict) -> Dict:
        """Generate insights from SAR analysis - supports all 10 analysis types."""
        insights = {
            'analysis_type': analysis_type,
            'summary': '',
            'key_findings': [],
            'recommendations': []
        }
        
        analysis_lower = analysis_type.lower()
        
        # 1. Oil Spill Detection
        if 'oil' in analysis_lower or 'spill' in analysis_lower:
            num_slicks = stats.get('num_detected_slicks', 0)
            area_km2 = stats.get('total_slick_area_km2', 0)
            coverage = stats.get('coverage_percent', 0)
            
            insights['summary'] = (
                f"Oil spill detection identified {num_slicks} potential slick(s) "
                f"covering {area_km2:.2f} km² ({coverage:.2f}% of area)."
            )
            
            if num_slicks > 0:
                insights['key_findings'].append(
                    f"⚠️ {num_slicks} potential oil slick(s) detected"
                )
                if area_km2 > 1.0:
                    insights['key_findings'].append(
                        f"Large spill area: {area_km2:.2f} km² - significant environmental concern"
                    )
                insights['recommendations'].append(
                    "Immediate verification with auxiliary data (optical, AIS, wind) recommended"
                )
                insights['recommendations'].append(
                    "Deploy response teams to confirmed spill locations"
                )
            else:
                insights['key_findings'].append("No significant oil slicks detected")
        
        # 2. Ship Detection
        elif 'ship' in analysis_lower:
            num_ships = stats.get('num_detected_ships', 0)
            
            insights['summary'] = f"Ship detection identified {num_ships} potential vessel(s)."
            
            if num_ships > 0:
                insights['key_findings'].append(f"{num_ships} vessel(s) detected in area")
                if 'largest_ship_m2' in stats:
                    insights['key_findings'].append(
                        f"Largest vessel: {stats['largest_ship_m2']:.0f} m²"
                    )
                insights['recommendations'].append(
                    "Cross-reference with AIS data for vessel identification"
                )
                if num_ships > 10:
                    insights['recommendations'].append(
                        "High vessel density - monitor for illegal fishing or congestion"
                    )
            else:
                insights['key_findings'].append("No ships detected in scene")
        
        # 3. Crop Monitoring
        elif 'crop' in analysis_lower:
            mean_rvi = stats.get('mean', 0)
            high_vigor = stats.get('high_vigor_percent', 0)
            veg_area = stats.get('vegetated_area_km2', 0)
            
            insights['summary'] = (
                f"Crop monitoring shows mean RVI of {mean_rvi:.3f}. "
                f"{high_vigor:.1f}% high vigor crops across {veg_area:.2f} km²."
            )
            
            if high_vigor > 50:
                insights['key_findings'].append(
                    f"Excellent crop health - {high_vigor:.1f}% high vigor"
                )
            elif high_vigor < 20:
                insights['key_findings'].append(
                    f"⚠️ Limited high-vigor crops ({high_vigor:.1f}%) - investigate causes"
                )
                insights['recommendations'].append(
                    "Check for water stress, nutrient deficiency, or pest damage"
                )
            
            if mean_rvi < 0.3:
                insights['recommendations'].append(
                    "Low vegetation index - consider irrigation or management intervention"
                )
        
        # 4. Land Cover Classification
        elif 'land cover' in analysis_lower or 'classification' in analysis_lower:
            num_classes = stats.get('num_classes', 0)
            
            insights['summary'] = f"Land cover classification identified {num_classes} distinct classes."
            
            class_info = []
            for i in range(num_classes):
                pct = stats.get(f'class_{i}_percent', 0)
                area = stats.get(f'class_{i}_area_km2', 0)
                class_info.append(f"Class {i}: {pct:.1f}% ({area:.2f} km²)")
            
            if class_info:
                insights['key_findings'].append("Class distribution:")
                insights['key_findings'].extend(class_info)
            
            insights['recommendations'].append(
                "Validate classification with ground truth data"
            )
        
        # 5. Biomass Estimation
        elif 'biomass' in analysis_lower:
            mean_idx = stats.get('mean', 0)
            forest_area = stats.get('forest_area_km2', 0)
            high_biomass = stats.get('high_biomass_percent', 0)
            
            insights['summary'] = (
                f"Biomass estimation shows mean index of {mean_idx:.3f}. "
                f"Forest area: {forest_area:.2f} km² with {high_biomass:.1f}% high biomass."
            )
            
            if high_biomass > 30:
                insights['key_findings'].append(
                    f"Significant mature forest - {high_biomass:.1f}% high biomass areas"
                )
            elif forest_area < 1:
                insights['key_findings'].append(
                    "Limited forest cover - deforested or non-forested landscape"
                )
            
            insights['recommendations'].append(
                "Calibrate with field measurements for absolute biomass (tons/ha)"
            )
        
        # 6. Wildfire Burn Mapping
        elif 'fire' in analysis_lower or 'burn' in analysis_lower:
            burned_area = stats.get('burned_area_km2', 0)
            burned_pct = stats.get('burned_percent', 0)
            high_sev = stats.get('high_severity_percent', 0)
            
            insights['summary'] = (
                f"Wildfire burn mapping detected {burned_area:.2f} km² burned "
                f"({burned_pct:.1f}% of area)."
            )
            
            if burned_area > 0:
                insights['key_findings'].append(
                    f"🔥 Burned area detected: {burned_area:.2f} km²"
                )
                if high_sev > 10:
                    insights['key_findings'].append(
                        f"⚠️ High severity burn: {high_sev:.1f}% of burned area"
                    )
                insights['recommendations'].append(
                    "Prioritize high-severity areas for restoration"
                )
                insights['recommendations'].append(
                    "Monitor for erosion and vegetation recovery"
                )
            else:
                insights['key_findings'].append("No significant burn areas detected")
        
        # 7. Geology/Terrain Analysis
        elif 'geology' in analysis_lower or 'terrain' in analysis_lower:
            mean_rough = stats.get('mean', 0)
            lineament_density = stats.get('lineament_density_percent', 0)
            mountain_area = stats.get('mountainous_area_km2', 0)
            
            insights['summary'] = (
                f"Terrain analysis shows mean roughness of {mean_rough:.3f}. "
                f"Lineament density: {lineament_density:.2f}%, Mountainous: {mountain_area:.2f} km²."
            )
            
            if mean_rough > 0.6:
                insights['key_findings'].append(
                    "Highly rugged terrain - mountains or steep slopes"
                )
            elif mean_rough < 0.3:
                insights['key_findings'].append(
                    "Smooth terrain - plains or low-relief landscape"
                )
            
            if lineament_density > 5:
                insights['key_findings'].append(
                    f"High lineament density ({lineament_density:.2f}%) - active geological structures"
                )
                insights['recommendations'].append(
                    "Investigate lineaments for fault zones or geological hazards"
                )
        
        # 8. Flood Mapping
        elif 'flood' in analysis_lower:
            water_pct = stats.get('water_percent', 0)
            water_area = stats.get('water_area_km2', 0)
            
            insights['summary'] = (
                f"Flood mapping detected {water_pct:.1f}% water coverage ({water_area:.2f} km²)."
            )
            
            if water_pct > 30:
                insights['key_findings'].append(
                    f"⚠️ Significant flooding detected ({water_pct:.1f}%)"
                )
                insights['recommendations'].append(
                    "Prioritize affected areas for emergency response"
                )
            elif water_pct > 10:
                insights['key_findings'].append(
                    f"Moderate water presence ({water_pct:.1f}%) - monitor situation"
                )
            else:
                insights['key_findings'].append(
                    f"Limited water extent ({water_pct:.1f}%)"
                )
        
        # 9. Polarimetric Analysis
        elif 'polarimetric' in analysis_lower or 'polari' in analysis_lower:
            forest_pct = stats.get('forest_percent', 0)
            urban_pct = stats.get('urban_percent', 0)
            agri_pct = stats.get('agricultural_percent', 0)
            
            insights['summary'] = (
                f"Polarimetric analysis: {forest_pct:.1f}% forest, "
                f"{urban_pct:.1f}% urban, {agri_pct:.1f}% agricultural."
            )
            
            if forest_pct > 50:
                insights['key_findings'].append(
                    f"Forest-dominated landscape ({forest_pct:.1f}%)"
                )
            if urban_pct > 30:
                insights['key_findings'].append(
                    f"Significant urban development ({urban_pct:.1f}%)"
                )
            if agri_pct > 40:
                insights['key_findings'].append(
                    f"Agricultural landscape ({agri_pct:.1f}%)"
                )
        
        # 10. Soil Moisture
        elif 'soil' in analysis_lower or 'moisture' in analysis_lower:
            mean_sm = stats.get('mean', 0)
            very_dry = stats.get('very_dry_percent', 0)
            moist = stats.get('moist_percent', 0) + stats.get('very_moist_percent', 0)
            
            insights['summary'] = (
                f"Soil moisture estimation shows mean index of {mean_sm:.3f}. "
                f"{very_dry:.1f}% very dry, {moist:.1f}% moist areas."
            )
            
            if very_dry > 40:
                insights['key_findings'].append(
                    f"⚠️ High drought stress - {very_dry:.1f}% very dry soils"
                )
                insights['recommendations'].append(
                    "Consider irrigation or drought management strategies"
                )
            elif moist > 50:
                insights['key_findings'].append(
                    f"Good soil moisture levels - {moist:.1f}% moist/very moist"
                )
            
            insights['recommendations'].append(
                "Validate with in-situ sensors for absolute moisture content"
            )
        
        # Generic SAR analysis (fallback)
        else:
            mean_val = stats.get('mean', 0)
            insights['summary'] = f"{analysis_type} analysis shows mean value of {mean_val:.3f}."
            insights['key_findings'].append("Analysis complete - review results")
        
        logger.info(f"Generated {analysis_type} insights")
        return insights
    
    @staticmethod
    def generate_change_detection_insights(change_data: np.ndarray, 
                                          stats: Dict) -> Dict:
        """Generate insights from change detection."""
        insights = {
            'analysis_type': 'Change Detection',
            'summary': '',
            'key_findings': [],
            'recommendations': []
        }
        
        changed_pct = stats.get('changed_percent', 0)
        
        insights['summary'] = (
            f"Change detection analysis identified {changed_pct:.1f}% of the "
            f"study area has undergone significant change."
        )
        
        if changed_pct > 40:
            insights['key_findings'].append(
                "Extensive changes detected - major land cover transformation"
            )
            insights['recommendations'].append(
                "Conduct detailed investigation of change drivers and patterns"
            )
        elif changed_pct > 20:
            insights['key_findings'].append(
                "Moderate changes observed - notable land cover modifications"
            )
        elif changed_pct > 5:
            insights['key_findings'].append(
                "Minor changes detected - localized land cover alterations"
            )
        else:
            insights['key_findings'].append(
                "Minimal changes - study area remains largely stable"
            )
        
        logger.info("Generated change detection insights")
        return insights
    
    @staticmethod
    def format_insights_report(insights: Dict, output_path: Path = None) -> str:
        """
        Format insights as text report.
        
        Args:
            insights: Insights dictionary
            output_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append(f"ANALYSIS INSIGHTS: {insights['analysis_type']}")
        report.append("=" * 70)
        report.append("")
        
        report.append("SUMMARY")
        report.append("-" * 70)
        report.append(insights['summary'])
        report.append("")
        
        if insights['key_findings']:
            report.append("KEY FINDINGS")
            report.append("-" * 70)
            for i, finding in enumerate(insights['key_findings'], 1):
                report.append(f"{i}. {finding}")
            report.append("")
        
        if insights['recommendations']:
            report.append("RECOMMENDATIONS")
            report.append("-" * 70)
            for i, rec in enumerate(insights['recommendations'], 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Saved insights report to {output_path}")
        
        return report_text
