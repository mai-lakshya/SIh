class RecommendationEngine:
    """
    Generates actionable recommendations based on risk assessment
    """
    
    def __init__(self):
        self.recommendation_templates = self._load_templates()
    
    def _load_templates(self):
        """Load recommendation templates"""
        return {
            'high_legal_risk': {
                'actions': [
                    'Conduct comprehensive legal audit',
                    'Engage specialized legal counsel',
                    'Fast-track dispute resolution mechanisms',
                    'Document all legal compliance measures'
                ],
                'priority': 'High',
                'timeframe': 'Immediate (0-30 days)',
                'expected_impact': 'Reduce delay risk by 25-40%'
            },
            'high_social_risk': {
                'actions': [
                    'Establish community liaison office',
                    'Conduct town hall meetings',
                    'Implement grievance redressal mechanism',
                    'Enhance compensation packages'
                ],
                'priority': 'High',
                'timeframe': 'Short-term (0-60 days)',
                'expected_impact': 'Reduce delay risk by 20-35%'
            },
            'clearance_delays': {
                'actions': [
                    'Identify bottleneck clearances',
                    'Engage with regulatory authorities',
                    'Prepare expedited documentation',
                    'Consider parallel processing'
                ],
                'priority': 'Medium-High',
                'timeframe': 'Medium-term (30-90 days)',
                'expected_impact': 'Reduce delay risk by 15-25%'
            },
            'financial_risk': {
                'actions': [
                    'Review budget allocation',
                    'Secure additional funding sources',
                    'Optimize disbursement schedule',
                    'Implement cost control measures'
                ],
                'priority': 'Medium',
                'timeframe': 'Short-term (0-45 days)',
                'expected_impact': 'Reduce delay risk by 15-20%'
            },
            'environmental_risk': {
                'actions': [
                    'Conduct additional environmental studies',
                    'Implement mitigation measures',
                    'Engage environmental agencies',
                    'Monitor compliance metrics'
                ],
                'priority': 'Medium',
                'timeframe': 'Medium-term (30-90 days)',
                'expected_impact': 'Reduce delay risk by 10-20%'
            }
        }
    
    def generate_recommendations(self, risk_drivers, project_metadata):
        """
        Generate prioritized recommendations
        """
        recommendations = []
        
        # Check each risk driver and add appropriate recommendations
        for feature, importance in risk_drivers:
            feature_lower = feature.lower()
            if 'dispute' in feature_lower or 'legal' in feature_lower:
                rec = self._create_recommendation('high_legal_risk', feature, importance)
                recommendations.append(rec)
            elif 'protest' in feature_lower or 'family' in feature_lower or 'families' in feature_lower:
                rec = self._create_recommendation('high_social_risk', feature, importance)
                recommendations.append(rec)
            elif 'clearance' in feature_lower or 'section_11' in feature_lower or 'p_r' in feature_lower or 'complexity' in feature_lower:
                rec = self._create_recommendation('clearance_delays', feature, importance)
                recommendations.append(rec)
            elif 'fund' in feature_lower or 'cost' in feature_lower or 'deficit' in feature_lower or 'gap' in feature_lower or 'age' in feature_lower or 'interaction' in feature_lower:
                rec = self._create_recommendation('financial_risk', feature, importance)
                recommendations.append(rec)
            elif 'environment' in feature_lower or 'terrain' in feature_lower or 'area' in feature_lower:
                rec = self._create_recommendation('environmental_risk', feature, importance)
                recommendations.append(rec)
            else:
                rec = self._create_recommendation('financial_risk', feature, importance)
                recommendations.append(rec)
        
        # Sort by priority
        priority_order = {'Critical': 0, 'High': 1, 'Medium-High': 2, 'Medium': 3, 'Low': 4}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 5))
        
        # Add project-specific recommendations
        recommendations.extend(self._get_project_specific_recommendations(project_metadata))
        
        return recommendations
    
    def _create_recommendation(self, template_key, feature, importance):
        """Create recommendation from template"""
        template = self.recommendation_templates.get(template_key, {})
        
        return {
            'issue': f"High risk from {feature.replace('_', ' ').title()}",
            'actions': template.get('actions', ['Investigate issue']),
            'priority': template.get('priority', 'Medium'),
            'timeframe': template.get('timeframe', 'Short-term'),
            'expected_impact': template.get('expected_impact', 'Varies'),
            'risk_driver': feature,
            'importance': importance,
            'template_key': template_key
        }
    
    def _get_project_specific_recommendations(self, metadata):
        """Generate project-specific recommendations"""
        recs = []
        
        # Check if it's a large project
        if metadata.get('estimated_cost_inr_crore', 0) > 5000:
            recs.append({
                'issue': 'Large project requiring special attention',
                'actions': [
                    'Establish project steering committee',
                    'Implement project monitoring dashboard',
                    'Conduct regular stakeholder meetings'
                ],
                'priority': 'High',
                'timeframe': 'Immediate',
                'expected_impact': 'Improve project oversight',
                'is_general': True
            })
        
        # Check if it's a forest area project
        if metadata.get('terrain_type') == 'Forest_Eco_Sensitive':
            recs.append({
                'issue': 'Forest area requires environmental compliance',
                'actions': [
                    'Ensure forest clearance compliance',
                    'Monitor environmental impact',
                    'Conduct environmental audits'
                ],
                'priority': 'High',
                'timeframe': 'Before project start',
                'expected_impact': 'Ensure environmental compliance',
                'is_general': True
            })
        
        return recs
    
    def format_recommendations_for_display(self, recommendations):
        """
        Format recommendations for human-readable display
        """
        output = "\n" + "="*70 + "\n"
        output += f"{'RECOMMENDATIONS':^70}\n"
        output += "="*70 + "\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            output += f"* Recommendation #{i}: {rec['issue']}\n"
            output += f"   Priority: {rec['priority']}\n"
            output += f"   Timeframe: {rec['timeframe']}\n"
            output += f"   Expected Impact: {rec.get('expected_impact', 'N/A')}\n"
            output += "   Actions:\n"
            
            for action in rec['actions']:
                output += f"      - {action}\n"
            
            output += "\n"
        
        return output

def get_dynamic_implementation_cost(template_key, project_cost):
    # Fixed baseline costs in INR
    cost_map = {
        'high_legal_risk': 15_00_000,
        'high_social_risk': 5_00_000,
        'clearance_delays': 25_00_000,
        'financial_risk': 10_00_000,
        'environmental_risk': 25_00_000,
    }
    
    base_cost = cost_map.get(template_key, 10_00_000)  # Default 10 Lakhs
    
    # Optional: Add a small size premium for mega-projects (> 5,000 Crores)
    # project_cost is assumed to be in pure INR (if not, we might need to adjust, but let's assume it's true INR based on the prompt)
    if project_cost > 50_00_00_00_000:  # 5000 Crores
        base_cost = base_cost * 1.3  # 30% premium
    elif project_cost > 100_00_00_00_000:  # 10,000 Crores
        base_cost = base_cost * 1.5
    
    return base_cost

def calculate_roi_for_recommendation(recommendation, project_cost, delay_cost_per_day, model=None, X_sample=None):
    """
    Calculate potential ROI for implementing a recommendation
    """
    if model is not None and X_sample is not None and 'risk_driver' in recommendation:
        # Data-driven simulation of intervention
        X_mitigated = X_sample.copy()
        feature = recommendation['risk_driver']
        
        # Apply hypothetical mitigation based on feature type (using realistic 20% reduction)
        if 'clearance' in feature:
            X_mitigated[feature] = X_mitigated[feature] * 0.8
        elif 'dispute' in feature or 'protest' in feature:
            X_mitigated[feature] = X_mitigated[feature] * 0.8
        elif 'deficit' in feature:
            X_mitigated[feature] = X_mitigated[feature] * 0.8
        elif 'cost' in feature:
            X_mitigated[feature] = X_mitigated[feature] * 0.8
        else:
            # General mitigation: reduce the feature's risk impact by 20%
            X_mitigated[feature] = X_mitigated[feature] * 0.8
            
        # Predict original vs mitigated delay days using the ensemble
        orig_delay = model.predict(X_sample)['predicted_delay_days'][0]
        new_delay = model.predict(X_mitigated)['predicted_delay_days'][0]
        
        estimated_delay_days_saved = max(0, orig_delay - new_delay)
    else:
        # Fallback to heuristics if model is not provided
        current_delay_risk = recommendation.get('importance', 0.5)
        potential_reduction = 0.3  # Assume 30% reduction
        estimated_delay_days_saved = current_delay_risk * potential_reduction * 180  # 180 days base
        
    cost_savings = estimated_delay_days_saved * delay_cost_per_day
    
    implementation_cost = get_dynamic_implementation_cost(
        recommendation.get('template_key', 'default'), 
        project_cost
    )
    
    if implementation_cost == 0:
        implementation_cost = 1 # Avoid division by zero
        
    roi = (cost_savings - implementation_cost) / implementation_cost * 100
    
    return {
        'estimated_delay_days_saved': estimated_delay_days_saved,
        'cost_savings': cost_savings,
        'implementation_cost': implementation_cost,
        'roi_percentage': roi,
        'payback_period_days': implementation_cost / (cost_savings / 180) if cost_savings > 0 else 0
    }
